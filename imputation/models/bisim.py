import torch
import torch.nn as nn
from torch.autograd import Variable
import sim

RNN_HID_SIZE = 64
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()
    def build(self):
        self.sim_f = sim.Model()
        self.sim_b = sim.Model()
    def forward(self, data):
        ret_f = self.sim_f(data, 'forward')
        ret_b = self.reverse(self.sim_b(data, 'backward'))
        ret = self.merge_ret(ret_f, ret_b)
        return ret
    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        print('loss f', loss_f.item())
        loss_b = ret_b['loss']
        print('loss b', loss_b.item())
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        print('loss c for fingerprint', loss_c.item())
        loss_d = self.get_consistency_loss(ret_f['decoded_y'], ret_b['decoded_y'])
        print('loss d for reference point', loss_d.item())
        loss = loss_f + loss_b + loss_c + loss_d
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        decoded_y = (ret_f['decoded_y'] + ret_b['decoded_y']) / 2
        ret_f['loss'] = loss
        ret_f['imputations'] = imputations
        ret_f['decoded_y'] = decoded_y
        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)
            if torch.cuda.is_available():
                indices = indices.cuda()
            return tensor_.index_select(1, indices)
        for key in ret:
            ret[key] = reverse_tensor(ret[key])
        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret

