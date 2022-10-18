import json
import os
import time
import scipy.stats
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import sys
import random
import argparse
pd.set_option('display.float_format', lambda x: '%.0f' % x)
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='akm')
parser.add_argument('--site', type=str, default='KDM')
parser.add_argument('--thre', type=str, default=0.1)
args = parser.parse_args()

random.seed(2021)
def parse_delta(masks, tss):
    deltas = []
    time_len, feat_len = masks.shape
    for ti in range(time_len):
        if ti == 0:
            deltas.append(np.ones(feat_len) * tss[ti])
        else:
            delta_ti = np.ones(feat_len) * tss[ti] + (1 - masks[ti - 1]) * deltas[-1]
            deltas.append(delta_ti)
    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, tss, dir_):
    rec_list = []
    deltas = parse_delta(masks, tss)
    for t in range(masks.shape[0]):
        rec = {}
        rec['values'] = values[t,:].tolist()
        rec['masks'] = masks[t,:].tolist()
        # imputation ground-truth
        rec['forwards'] = 0
        rec['deltas'] = deltas[t,:].tolist()
        rec['eval_masks'] = eval_masks[t,:].tolist()
        rec['evals'] = evals[t,:].tolist()
        rec_list.append(rec)
    return rec_list

def find_waypoint(t, wp_df):
    wp_df = wp_df.sort_values(by=['wp_ts'], ascending=True)
    wp_df_a = wp_df.shift(-1).rename(columns={'wp_ts':'a_ts', 'x':'a_x', 'y':'a_y'})
    wp_df_c = pd.concat([wp_df, wp_df_a], axis=1)
    target_row = wp_df_c.loc[(wp_df_c['wp_ts']<=t)&(wp_df_c['a_ts']>t),['wp_ts','x','y','a_ts', 'a_x', 'a_y']].reset_index(drop=True)
    if not target_row.empty:
        x = target_row.loc[0,'x'] + ((target_row.loc[0,'a_x']-target_row.loc[0,'x'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        y = target_row.loc[0,'y'] + ((target_row.loc[0,'a_y']-target_row.loc[0,'y'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        # print('target_row', x, y)
        return x, y
    else:
        wp_df = wp_df.drop_duplicates(subset=['wp_ts', 'x', 'y'])
        wp_df = wp_df.reset_index(drop=True)
        if len(wp_df)>=2:
            if t <= wp_df.loc[0,'wp_ts']:
                # print('t',t)
                # print('wp_df', wp_df)
                x = wp_df.loc[0,'x'] - ((wp_df.loc[1, 'x'] - wp_df.loc[0, 'x']) * (wp_df.loc[0,'wp_ts'] - t))/(wp_df.loc[1, 'wp_ts'] - wp_df.loc[0, 'wp_ts'])
                y = wp_df.loc[0,'y'] - ((wp_df.loc[1, 'y'] - wp_df.loc[0, 'y']) * (wp_df.loc[0,'wp_ts'] - t))/(wp_df.loc[1, 'wp_ts'] - wp_df.loc[0, 'wp_ts'])
                # print('approach 111', x,y)
            else:
                x = ((wp_df.loc[len(wp_df)-1,'x'] - wp_df.loc[len(wp_df)-2,'x'])*(t - wp_df.loc[len(wp_df)-1,'wp_ts']))/(wp_df.loc[len(wp_df)-1,'wp_ts'] - wp_df.loc[len(wp_df)-2,'wp_ts']) +  wp_df.loc[len(wp_df)-1,'x']
                y = ((wp_df.loc[len(wp_df)-1,'y'] - wp_df.loc[len(wp_df)-2,'y'])*(t - wp_df.loc[len(wp_df)-1,'wp_ts']))/(wp_df.loc[len(wp_df)-1,'wp_ts'] - wp_df.loc[len(wp_df)-2,'wp_ts']) +  wp_df.loc[len(wp_df)-1,'y']
                # print('approach 222', x, y)
        else:
            if t <= wp_df.loc[0,'wp_ts']:
                x = wp_df.loc[0,'x'] - (wp_df.loc[0,'wp_ts']-t)*1
                y = wp_df.loc[0,'y'] - (wp_df.loc[0,'wp_ts']-t)*1
                # print('approach 333', x, y)
            else:
                x = (t - wp_df.loc[0,'wp_ts'])*1 + wp_df.loc[0,'x']
                y = (t - wp_df.loc[0,'wp_ts'])*1 + wp_df.loc[0,'y']
                # print('approach 444', x, y)
        return x, y

#linear interpolatinig RP as intial values
def interpolate_rp(group_df):
    ori_shape = group_df.shape
    ori_index = group_df.index
    have_wp_df = group_df.loc[~group_df['wp_ts'].isnull(), :]
    # print('have_wp_df shape', have_wp_df.shape)
    wp_df = have_wp_df.loc[:, ['wp_ts', 'x', 'y']]
    have_null_wp_df = group_df.loc[group_df['wp_ts'].isnull(), :]
    # print('have_null_wp_df shape', have_null_wp_df.shape)
    if not have_null_wp_df.empty:
        for index, row in have_null_wp_df.iterrows():
            ts = int(row['ts'])
            x, y = find_waypoint(ts, wp_df)
            if x and y:
                have_null_wp_df.loc[index, 'x'] = x
                have_null_wp_df.loc[index, 'y'] = y
                have_null_wp_df.loc[index, 'wp_ts'] = ts
            else:
                print('Hahahahah')
    group_df = pd.concat([have_wp_df, have_null_wp_df], axis=0)
    group_df = group_df.loc[ori_index,:]
    post_shape = group_df.shape
    assert (ori_shape[0]==post_shape[0]) & (ori_shape[1]==post_shape[1]), 'error'
    ip_x_y = group_df.loc[:, ['x', 'y']]
    return ip_x_y

# generate mask vectors for fingerprint and RP respectively, and reserve some data as  groundtruth
def transform_sample(sample_df, xy_df):
    feats = sample_df.drop(['floor', 'x', 'y', 'wp_ts', 'ts', 'path'], axis=1)
    tss = sample_df['ts'].diff().fillna(0).map(lambda x: round(x / 1000, 0)).values.tolist()
    tss_bk = sample_df.iloc[::-1]['ts'].diff().abs().fillna(0).map(lambda x: round(x / 1000, 0)).values.tolist()
    # print('feats', feats.iloc[0,:10])
    labels_evals = sample_df.loc[:, ['x', 'y']].values
    label_shp = labels_evals.shape
    # mark the groundtruth for RPs
    indices_y = np.unique(np.where(~np.isnan(labels_evals))[0])
    cnt = 0
    if len(indices_y)>=4:
        man_indices_y = np.random.choice(indices_y, 1)
        cnt+=1
    elif len(indices_y)>=2 and len(indices_y)<4:
        cnt+=1
        man_indices_y = np.random.choice(indices_y, 1)
    elif len(indices_y)>=1 and len(indices_y)<2:
        rn = random.random()
        if rn < 0.12:
            cnt+=1
            man_indices_y = np.random.choice(indices_y, 1)
        else:
            man_indices_y = False
    else:
        man_indices_y = False
    # print('man_indices_y', man_indices_y)
    labels_values = labels_evals.copy()
    labels_values[man_indices_y] = np.nan
    masks_y = ~np.isnan(labels_values)
    eval_masks_y = (~np.isnan(labels_values)) ^ (~np.isnan(labels_evals))
    delta_y = parse_delta(masks_y.astype(int), tss)
    masks_y = masks_y.astype(int).tolist()
    eval_masks_y = eval_masks_y.astype(int).tolist()
    labels_values = np.nan_to_num(labels_values, nan=0)
    labels_values = (labels_values - mean_x_y)/std_x_y
    labels_evals = np.nan_to_num(labels_evals, nan=0)
    labels_evals = (labels_evals - mean_x_y)/std_x_y

    labels_values = np.round(labels_values, 6)
    labels_list = labels_values.tolist()
    labels_evals = np.round(labels_evals, 6)
    labels_eval_list = labels_evals.tolist()

    evals = feats.values.astype(float)
    shp = evals.shape
    evals = evals.reshape(-1)
    # randomly eliminate 10% values as the imputation ground-truth
    # indices = np.where((~np.isnan(evals))&(evals!=-100))[0].tolist()
    # indices = np.random.choice(indices, len(indices) // 10)
    values = evals.copy()
    #values[indices] = np.nan
    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))
    evals = evals.reshape(shp)
    values = values.reshape(shp)
    values = np.nan_to_num(values, nan=-100)
    values = (values - mean) / std
    evals = np.nan_to_num(evals, nan=-100)
    evals = (evals - mean) / std
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    values = np.round(values, 6)
    evals = np.round(evals, 6)
    masks = masks.astype(int)
    eval_masks = eval_masks.astype(int)
    rec = {'label': labels_list}
    rec['masks_y'] = masks_y
    rec['delta_y'] = delta_y.tolist()
    rec['eval_masks_y'] = eval_masks_y
    rec['eval_label'] = labels_eval_list
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, tss, dir_='forward')
    rec['backward'] = parse_rec(values[::-1],masks[::-1], evals[::-1], eval_masks[::-1], tss_bk, dir_='backward')
    return rec, len(indices_y), cnt

#load the differentiated results of original data
data_root_path = '../data'
data_path = os.path.join(data_root_path, args.site)
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_filterd_{site}_{method}_{thre}.csv'.format(site=args.site, method=args.method, thre=args.thre)), header=0)


other_columns = ['floor', 'x', 'y', 'wp_ts','ts', 'path']
# outlier columns removing
all_null_df = pd.DataFrame(pd.isnull(wifi_df[wifi_df==-100]).sum(axis=0))
all_null_cols = list(all_null_df[all_null_df[0]==0].index)
wifi_df = wifi_df.drop(all_null_cols, axis=1)
out_loc = np.where(wifi_df.values=='-1-100.0')

mean = wifi_df.drop(other_columns, axis=1).astype(float).mean().values
std = wifi_df.drop(other_columns, axis=1).astype(float).std().values

mean_x_y = wifi_df.loc[:,['x','y']].mean().values
std_x_y = wifi_df.loc[:,['x','y']].std().values

wifi_df_b = wifi_df.copy()
seq_len = 5
path_groups = wifi_df_b.groupby(['path'])
print('len of paths', len(path_groups))

path_len = []
path_len_list = []
with open(os.path.join(data_path, 'wifi_biseq_{method}_{thre}.json'.format(method=args.method, thre=args.thre)), 'w+') as file:
    cnt = 0
    sum_y_labels = 0
    sum_eval_y_labels = 0
    for name, group in path_groups:
        group_df = pd.DataFrame(group)
        path_len_list.append(len(group_df))
        group_df = group_df.sort_values(by='ts')
        if len(group_df) >= seq_len:
            group_df_cp = group_df.copy()
            ip_x_y = interpolate_rp(group_df_cp)
            for i in range(0, len(group_df)-seq_len):
                group_df_cp = group_df.iloc[i:i+seq_len, :]
                # print('group_df_cp shape', group_df_cp.shape)
                xy_df = ip_x_y.iloc[i:i+seq_len, :]
                # print('xy_df shape', xy_df.shape)
                one_sample, len_y_labels, len_eval_y_labels  = transform_sample(group_df_cp, xy_df)
                # one_sample = transform_sample(group_df_cp)
                sum_y_labels = sum_y_labels + len_y_labels
                sum_eval_y_labels = sum_eval_y_labels + len_eval_y_labels
                rec = json.dumps(one_sample)
                file.write(rec + '\n')
                cnt += 1



