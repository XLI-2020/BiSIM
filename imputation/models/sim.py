import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import argparse
import copy
SEQ_LEN = 5
RNN_HID_SIZE = 64
FEATURE_LEN = 923#665
ENC_HID_SIZE = 923#665

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 1) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 1]

        #energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(FEATURE_LEN*2, RNN_HID_SIZE)
        #self.decoder_cell = nn.LSTMCell(2, RNN_HID_SIZE)
        self.decoder_cell = nn.LSTMCell(ENC_HID_SIZE+2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = FEATURE_LEN, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = FEATURE_LEN, output_size = FEATURE_LEN, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, FEATURE_LEN)
        self.feat_reg = FeatureRegression(FEATURE_LEN)
        self.ATT = Attention(enc_hid_dim=ENC_HID_SIZE, dec_hid_dim=RNN_HID_SIZE)
        self.enc_fc = nn.Linear(RNN_HID_SIZE, ENC_HID_SIZE)
        self.weight_combine = nn.Linear(FEATURE_LEN*2, FEATURE_LEN)

        self.dropout = nn.Dropout(p = 0.25)
        # self.out = nn.Linear(RNN_HID_SIZE, 1)
        self.out = nn.Linear(RNN_HID_SIZE, 2)

    def reverse_tensor(self,tensor_):
        if tensor_.dim() <= 1:
            return tensor_
        indices = range(tensor_.size()[1])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad=False)
        if torch.cuda.is_available():
            indices = indices.cuda()
        return tensor_.index_select(1, indices)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels']
        masks_yy = data['masks_y']
        eval_masks_y = data['eval_masks_y']
        eval_label = data['eval_label']
        if direct == 'forward':
            labels = self.reverse_tensor(labels)
            masks_yy = self.reverse_tensor(masks_yy)
            eval_masks_y = self.reverse_tensor(eval_masks_y)
            eval_label = self.reverse_tensor(eval_label)
        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0
        imputations = []
        y_decoder_list = []
        enc_hid_output = []
        #Encoder part
        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            gamma_h = self.temp_decay_h(d)
            # print('h ori', h.shape)
            h_ori = copy.copy(h)
            h = h * gamma_h
            x_h = self.hist_reg(h)
            # x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            x_loss += F.mse_loss(x*m, x_h*m)/ (torch.sum(m) + 1e-5)
            x_c =  m * x + (1 - m) * x_h
            enc_hid = torch.tanh(self.enc_fc(h))
            enc_hid = enc_hid*m
            #enc_hid_output.append(h_ori.unsqueeze(1)) #put decayed hiddened state into enc hid list
            enc_hid_output.append(enc_hid.unsqueeze(1)) #put decayed hiddened state into enc hid list
            inputs = torch.cat([x_c, m], dim = 1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(x_c.unsqueeze(dim = 1))
        y_h = self.out(h)
        enc_hid_output = torch.cat(enc_hid_output, dim=1)
        temp_y_loss = F.mse_loss(y_h*masks_yy[:, 0, :], labels[:,0,:]* masks_yy[:, 0, :])/(torch.sum(masks_yy[:, 0, :]) + 1e-5)
        y_loss += temp_y_loss
        input_0 = labels[:,0,:]* masks_yy[:, 0, :] + y_h*(1-masks_yy[:, 0, :])
        y_decoder_list.append(input_0.unsqueeze(dim = 1))
        #Decoder part
        for td in range(1,SEQ_LEN):
            label_y = labels[:, td, :]
            mask_y = masks_yy[:, td, :]
            input_y = label_y*mask_y + y_h*(1-mask_y)
            attn = self.ATT(h, enc_hid_output) # attn = [batch size, src len]
            # print('attn shape', attn.shape)
            attn = attn.unsqueeze(1) # attn = [batch size, 1, src len]
            context = torch.bmm(attn, enc_hid_output) # context=[batch size, 1, enc hid dim]
            context = context.squeeze(1)
            input_y_com = torch.cat([input_y, context], axis=1)
            # print('combined input_y shape', input_y_com.shape)
            h, c = self.decoder_cell(input_y_com, (h, c))
            y_h = self.out(h)
            temp_y_loss = F.mse_loss(y_h*mask_y, label_y* mask_y)/(torch.sum(mask_y) + 1e-5)
            y_loss += temp_y_loss
            y_decoder_list.append(input_y.unsqueeze(dim = 1))
        imputations = torch.cat(imputations, dim = 1)
        decoded_y = torch.cat(y_decoder_list, dim=1)
        # reverse deocded result and labels to be consistent with imputations
        decoded_y = self.reverse_tensor(decoded_y)
        labels = self.reverse_tensor(labels)
        masks_yy = self.reverse_tensor(masks_yy)
        eval_masks_y = self.reverse_tensor(eval_masks_y)
        eval_label = self.reverse_tensor(eval_label)
        # print((masks_yy*decoded_y)[0,0,0],  (masks_yy*labels)[0,0,0])
        assert (masks_yy*decoded_y)[0,0,0] == (masks_yy*labels)[0,0,0], 'error unaligned'
        imputations = torch.round(imputations*(10**6))/(10**6)
        decoded_y = torch.round(decoded_y*(10**6))/(10**6)
        labels = torch.round(labels*(10**6))/(10**6)

        # print(direct)
        # print('x_loss', x_loss.item() / SEQ_LEN)
        # print('y_loss', y_loss.item()/SEQ_LEN)
        return {'loss': (x_loss / SEQ_LEN) + (y_loss/SEQ_LEN),
                'imputations': imputations, 'labels': labels, 'decoded_y':decoded_y, 'masks_y':masks_yy, 'evals':evals,
                'eval_masks':eval_masks, 'eval_masks_y':eval_masks_y, 'eval_label':eval_label}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
