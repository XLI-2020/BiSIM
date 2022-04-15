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
from baselines import BISEQ_evaluate, inver_trans_output, BISEQ_localization
from baselines.Ours import data_loader, models
import pandas as pd
import ujson as json
from sklearn import metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--model', type = str, default='brits')
args = parser.parse_args()

site = 'site3'
floor_num = 'F1'
derived_data_path = '../../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)
brits_data_path = os.path.join(derived_data_path, 'biseq')

train_epoch_loss = []
inner_local_dist = []
def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_iter = data_loader.get_loader(batch_size = args.batch_size)
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)
            run_loss += ret['loss'].item()
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),)
        train_epoch_loss.append(run_loss / (idx + 1.0))
        val_data_iter = data_loader.get_val_loader(batch_size = args.batch_size)
        if epoch % 1 == 0:
            evaluate(model, data_iter, val_data_iter)
        print('evolving...')
        print(train_epoch_loss)
        print(inner_local_dist)

def evaluate(model, val_iter, eval_iter):
    model.eval()
    labels = []
    imputations = []
    decoded_y_list = []
    y_masks = []
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        # label = label.reshape(-1,2)

        masks_y = ret['masks_y'].data.cpu().numpy()
        y_masks += masks_y.tolist()
        is_train = ret['is_train'].data.cpu().numpy()

        # eval_masks = ret['eval_masks'].data.cpu().numpy()
        # eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        # evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation.tolist()

        decoded_y = ret['decoded_y'].data.cpu().numpy()
        decoded_y_list += decoded_y.tolist()
        # collect test label & prediction
        # pred = pred[np.where(is_train == 0)]
        # label = label[np.where(is_train == 0)]
        labels += label.tolist()
        # preds += pred.tolist()

    labels_y = np.array(labels)
    print('labels', labels_y.shape)

    masks_y = np.array(y_masks)
    print('masks_y', masks_y.shape)

    # evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    print('imputation', imputations.shape)
    decoded_y = np.asarray(decoded_y_list)
    print('decoded_y', decoded_y.shape)

    # labels_y = labels_y.reshape((-1, labels_y.shape[2])).tolist()
    # radio_map = imputations.reshape((-1, imputations.shape[2])).tolist()
    # reference_points = decoded_y.reshape((-1, decoded_y.shape[2])).tolist()

    batch_size, seq_len, feat_len = imputations.shape
    labels_y = labels_y[:,-1,:]
    masks_y = masks_y[:,-1,:]
    radio_map = imputations[:,-1,:]
    reference_points = decoded_y[:,-1,:]

    # labels_y = labels_y.reshape((-1,2))
    # masks_y = masks_y.reshape((-1, 2))
    # radio_map = imputations.reshape(-1, feat_len)
    # reference_points = decoded_y.reshape(-1,2)

    mean, std, mean_x_y, std_x_y = inver_trans_output.get_mean_var_db()

    radio_map = radio_map*std + mean

    labels_y = labels_y*std_x_y + mean_x_y
    reference_points = reference_points*std_x_y + mean_x_y

    radio_map = radio_map.tolist()
    labels_y = labels_y.tolist()
    reference_points = reference_points.tolist()
    masks_y = masks_y.tolist()
    with open(os.path.join(brits_data_path, 'radio_map.json'), 'w+') as file:
        json.dump(radio_map, file)
    with open(os.path.join(brits_data_path, 'reference_points.json'), 'w+') as file:
        json.dump(reference_points, file)
    with open(os.path.join(brits_data_path, 'labels_y.json'), 'w+') as file:
        json.dump(labels_y, file)
    with open(os.path.join(brits_data_path, 'masks_y.json'), 'w+') as file:
        json.dump(masks_y, file)


    ### evaluation....
    labels = []
    imputations = []
    decoded_y_list = []
    y_masks = []
    for e_idx, e_data in enumerate(eval_iter):
        e_data = utils.to_var(e_data)
        ret = model.run_on_batch(e_data, None)
        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        # label = label.reshape(-1,2)

        masks_y = ret['masks_y'].data.cpu().numpy()
        y_masks += masks_y.tolist()
        is_train = ret['is_train'].data.cpu().numpy()

        # eval_masks = ret['eval_masks'].data.cpu().numpy()
        # eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        # evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation.tolist()

        decoded_y = ret['decoded_y'].data.cpu().numpy()
        decoded_y_list += decoded_y.tolist()
        # collect test label & prediction
        # pred = pred[np.where(is_train == 0)]
        # label = label[np.where(is_train == 0)]
        labels += label.tolist()
        # preds += pred.tolist()

    labels_y_e = np.array(labels)
    print('labels e', labels_y_e.shape)

    masks_y_e = np.array(y_masks)
    print('masks_y e', masks_y_e.shape)

    # evals = np.asarray(evals)
    imputations_e = np.asarray(imputations)
    print('imputation e', imputations_e.shape)
    decoded_y_e = np.asarray(decoded_y_list)
    print('decoded_y e', decoded_y_e.shape)

    # labels_y = labels_y.reshape((-1, labels_y.shape[2])).tolist()
    # radio_map = imputations.reshape((-1, imputations.shape[2])).tolist()
    # reference_points = decoded_y.reshape((-1, decoded_y.shape[2])).tolist()

    batch_size, seq_len, feat_len = imputations_e.shape
    labels_y_e = labels_y_e[:, -1, :]
    print('labels_y_e', labels_y_e.shape)
    masks_y_e = masks_y_e[:, -1, :]
    print('masks_y_e', masks_y_e.shape)

    fingerprint = imputations_e[:, -1, :]
    print('fingerprint', fingerprint.shape)

    fingerprint = fingerprint * std + mean
    labels_y_e = labels_y_e * std_x_y + mean_x_y

    ground_points = labels_y_e[np.unique(np.where(masks_y_e>0)[0])]
    ground_fps = fingerprint[np.unique(np.where(masks_y_e>0)[0])]
    print('ground fps, points', ground_fps.shape, ground_points.shape)

    evaluted_res = BISEQ_evaluate.get_online_evaluation_res(ground_fps, ground_points)
    print('inner localization distance:', evaluted_res)
    inner_local_dist.append(evaluted_res)

def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
