import torch
import torch.optim as optim
import numpy as np
import utils
import argparse
import sys
sys.path.append('/Users/xiaol/PycharmProjects/WifiLocalization')
from baselines import BISEQ_evaluate, inver_trans_output
from baselines.Ours import data_loader, models
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
fp_mae_list = []
fp_mre_list = []
rp_mae_list = []
rp_mre_list = []

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
        # val_data_iter = data_loader.get_val_loader(batch_size = args.batch_size)
        if epoch % 1 == 0:
            evaluate(model, data_iter)
        print('evolving...')
        print('loss:')
        print(train_epoch_loss)
        print('localization dist:')
        print(inner_local_dist)
        print('fp mae:')
        print(fp_mae_list)
        print('rp mae:')
        print(rp_mae_list)
        print('fp mre:')
        print(fp_mre_list)
        print('rp mre:')
        print(rp_mre_list)

def evaluate(model, val_iter):
    model.eval()
    labels = []
    evals = []
    imputations = []
    decoded_y_list = []
    y_masks = []
    eval_label = []
    mean, std, mean_x_y, std_x_y = inver_trans_output.get_mean_var_db()
    pos_imputations = []
    pos_decoded_y = []
    pos_eval_masks_y = []
    pos_eval_label = []
    eval_label_points = []
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        label = ret['labels'].data.cpu().numpy()
        masks_y = ret['masks_y'].data.cpu().numpy()
        y_masks += masks_y.tolist()
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        pos_imputations += imputation.tolist()

        print('imputations', len(imputations), len(evals))
        eval_masks_y = ret['eval_masks_y'].data.cpu().numpy()
        pos_eval_masks_y += eval_masks_y.tolist()

        eval_label_ = ret['eval_label'].data.cpu().numpy()
        eval_label += eval_label_[np.where(eval_masks_y == 1)].tolist()

        eval_batch_index, eval_seq_index, eval_label_index = np.where(eval_masks_y == 1)
        eval_label_points += eval_label_[np.unique(eval_batch_index), np.unique(eval_seq_index), :]
        print('eval_label_points', len(eval_label_points))

        pos_eval_label += eval_label_.tolist()

        decoded_y = ret['decoded_y'].data.cpu().numpy()
        decoded_y_list += decoded_y[np.where(eval_masks_y == 1)].tolist()
        pos_decoded_y += decoded_y.tolist()

        print('decoded_y_list', len(decoded_y_list), len(eval_label))
        labels += label.tolist()


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    eval_label = np.asarray(eval_label)
    decoded_y_list = np.asarray(decoded_y_list)

    fp_mae = np.abs(evals - imputations).mean()
    fp_mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    rp_mae = np.abs(eval_label - decoded_y_list).mean()
    rp_mre = np.abs(eval_label - decoded_y_list).sum() / np.abs(eval_label).sum()
    print('fp_mae', fp_mae)
    print('fp_mre', fp_mre)

    print('rp_mae', rp_mae)
    print('rp_mre', rp_mre)

    labels_y = np.array(labels)
    print('labels', labels_y.shape)

    pos_imputations = np.asarray(pos_imputations)
    print('pos_imputations', pos_imputations.shape)

    pos_decoded_y = np.asarray(pos_decoded_y)
    print('pos_decoded_y', pos_decoded_y.shape)

    pos_eval_masks_y = np.asarray(pos_eval_masks_y)
    print('pos_eval_masks_y', pos_eval_masks_y.shape)

    pos_eval_label = np.asarray(pos_eval_label)
    print('pos_eval_label', pos_eval_label.shape)

    pos_imputations = pos_imputations[:,-1,:]
    pos_decoded_y = pos_decoded_y[:,-1,:]
    pos_eval_masks_y = pos_eval_masks_y[:,-1,:]
    pos_eval_label = pos_eval_label[:,-1,:]

    all_index = list(range(pos_imputations.shape[0]))
    test_index = np.unique(np.where(pos_eval_masks_y==1)[0])
    train_index = [i for i in all_index if i not in test_index]
    print('test_index' ,len(test_index))

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

    # with open(os.path.join(brits_data_path, 'radio_map.json'), 'w+') as file:
    #     json.dump(radio_map, file)
    # with open(os.path.join(brits_data_path, 'reference_points.json'), 'w+') as file:
    #     json.dump(reference_points, file)
    # with open(os.path.join(brits_data_path, 'labels_y.json'), 'w+') as file:
    #     json.dump(labels_y, file)
    # with open(os.path.join(brits_data_path, 'masks_y.json'), 'w+') as file:
    #     json.dump(masks_y, file)

    evaluted_res = BISEQ_evaluate.get_localization_dist(radio_map, reference_points, fingprint_sample, ground_points)
    print('inner localization distance:', evaluted_res)
    inner_local_dist.append(evaluted_res)
    fp_mae_list.append(fp_mae)
    fp_mre_list.append(fp_mre)
    rp_mae_list.append(rp_mae)
    rp_mre_list.append(rp_mre)


def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
