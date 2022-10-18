import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import utils
import argparse
import sys
import data_loader, models
import os
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default=500)
parser.add_argument('--batch_size', type = int, default=32)
parser.add_argument('--site', type = str, default='KDM')
parser.add_argument('--method', type = str, default='akm')
parser.add_argument('--thre', type = float, default=0.1)
# parser.add_argument('--decay', type = str, default='en')
# parser.add_argument('--density', type = str, default='0')

args = parser.parse_args()

site = args.site
floor_num = args.floor

train_epoch_loss = []
inner_local_dist = []
fp_mae_list = []
rp_mae_list = []
w_inner_local_dist = []

p_rp_dist_list = []
rp_dist_list = []
#training model...
def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = data_loader.get_loader(batch_size = args.batch_size, method=args.method, thre=args.thre)
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)
            run_loss += ret['loss'].item()
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),)
        train_epoch_loss.append(round(run_loss / (idx + 1.0), 6))
        if epoch % 1 == 0:
            evaluate(model, data_iter)
        print('evolving...')
        print('loss:')
        print(train_epoch_loss)
        # print('localization dist:')
        # print(inner_local_dist)

        print('fp mae:')
        print(fp_mae_list)
        # print('rp dist:')
        # print(rp_dist_list)
        print('min localization dist: knn, wknn', min(inner_local_dist), min(w_inner_local_dist))
        # print('min rp dist:', min(rp_dist_list))

# evaluate the MAE, Euclidean distance of RPs
def evaluate(model, val_iter):
    model.eval()
    labels = []
    evals = []
    imputations = []
    decoded_y_list = []
    y_masks = []
    eval_label = []
    mean, std, mean_x_y, std_x_y = utils.get_mean_var_db(args.method, site, floor_num, args.thre)
    pos_imputations = []
    pos_decoded_y = []
    pos_eval_masks_y = []
    pos_eval_label = []
    eval_label_points = []
    decoded_y_points = []
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        label = ret['labels'].data.cpu().numpy()
        masks_y = ret['masks_y'].data.cpu().numpy()
        y_masks += masks_y.tolist()
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        imputation = imputation*std + mean
        eval_ = eval_*std + mean
        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        pos_imputations += imputation.tolist()
        # print('imputations', len(imputations), len(evals))
        eval_masks_y = ret['eval_masks_y'].data.cpu().numpy()
        pos_eval_masks_y += eval_masks_y.tolist()
        eval_label_ = ret['eval_label'].data.cpu().numpy()
        eval_label += eval_label_[np.where(eval_masks_y == 1)].tolist()
        pos_eval_label += eval_label_.tolist()
        eval_batch_index, eval_seq_index, eval_label_index = np.where(eval_masks_y == 1)
        # print(eval_batch_index,eval_seq_index,eval_label_index)
        eval_label_points += eval_label_[eval_batch_index[::2], eval_seq_index[::2], :].tolist()
        decoded_y = ret['decoded_y'].data.cpu().numpy()
        decoded_y_list += decoded_y[np.where(eval_masks_y == 1)].tolist()
        pos_decoded_y += decoded_y.tolist()
        decode_batch_index, decode_seq_index, decode_label_index = np.where(eval_masks_y == 1)
        decoded_y_points += decoded_y[decode_batch_index[::2], decode_seq_index[::2], :].tolist()
        # print('decoded_y_list', len(decoded_y_list), len(eval_label))
        labels += label.tolist()

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    eval_label = np.asarray(eval_label)
    decoded_y_list = np.asarray(decoded_y_list)

    eval_label_points = np.asarray(eval_label_points)
    decoded_y_points = np.asarray(decoded_y_points)
    eval_label_points = eval_label_points*std_x_y + mean_x_y
    decoded_y_points = decoded_y_points*std_x_y + mean_x_y
    sqrt_dist = np.sqrt(np.sum(np.square(eval_label_points-decoded_y_points),axis=1))
    # print('shape of sqrt_dist', sqrt_dist.shape)
    # print('points recover', np.sum(sqrt_dist)/len(sqrt_dist))
    rp_dist_list.append(round(np.sum(sqrt_dist)/len(sqrt_dist), 6))

    fp_mae = np.abs(evals - imputations).mean()

    labels_y = np.array(labels)
    pos_imputations = np.asarray(pos_imputations)
    pos_decoded_y = np.asarray(pos_decoded_y)
    pos_eval_masks_y = np.asarray(pos_eval_masks_y)
    pos_eval_label = np.asarray(pos_eval_label)
    pos_imputations = pos_imputations[:,-1,:]
    pos_decoded_y = pos_decoded_y[:,-1,:]
    pos_eval_masks_y = pos_eval_masks_y[:,-1,:]
    pos_eval_label = pos_eval_label[:,-1,:]
    bt_index = np.unique(np.where(pos_eval_masks_y==1)[0])
    points_eval_label = pos_eval_label[bt_index]
    points_eval_decode_y = pos_decoded_y[bt_index]
    points_eval_label = points_eval_label*std_x_y + mean_x_y
    points_eval_decode_y = points_eval_decode_y*std_x_y + mean_x_y
    p_sqrt_dist = np.sqrt(np.sum(np.square(points_eval_decode_y-points_eval_label),axis=1))
    p_rp_dist_list.append(round(np.sum(p_sqrt_dist)/len(p_sqrt_dist), 6))

    all_index = list(range(pos_imputations.shape[0]))
    test_index = np.unique(np.where(pos_eval_masks_y==1)[0])
    train_index = [i for i in all_index if i not in test_index]
    radio_map = pos_imputations[train_index]
    reference_points = pos_decoded_y[train_index]

    fingprint_sample = pos_imputations[test_index]
    ground_points = pos_eval_label[test_index]
    radio_map = radio_map*std + mean
    fingprint_sample = fingprint_sample*std + mean
    reference_points = reference_points*std_x_y + mean_x_y
    ground_points = ground_points*std_x_y + mean_x_y

    radio_map = radio_map.tolist()
    fingprint_sample = fingprint_sample.tolist()
    reference_points = reference_points.tolist()
    ground_points = ground_points.tolist()
    evaluted_res, wknn_dist = utils.get_localization_dist(radio_map, reference_points, fingprint_sample, ground_points)
    # print('inner localization distance:', evaluted_res, wknn_dist)
    inner_local_dist.append(evaluted_res)
    w_inner_local_dist.append(wknn_dist)
    fp_mae_list.append(round(fp_mae, 6))


def run():
    model = getattr(models, 'bisim').Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
