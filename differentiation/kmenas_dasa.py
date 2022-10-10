import os
import random
import time
import sys
sys.path.append('/Users/xiaol/PycharmProjects/WifiLocalization')
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from geo_analyze import geo_referencing,plot_points, plot_shape
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.cm as cm
import copy
from hc_custom_utils import Hierarchical, interpolate_rp

def calculate_fpr_tpr(threshold=0.1, p_eta=0.1, cluster=40, prop=1):
    random.seed(2021)
    data_root_path = '../data'
    site = 'KDM'
    if site == 'WDS':
        x_max, x_min = 40, 22
        y_max, y_min = 95, 75
    elif site == 'KDM':
        x_max, x_min = 80, 60
        y_max, y_min = 40, 20

    method = 'akm'
    p_eta = p_eta
    data_path = os.path.join(data_root_path, site)
    floor_df = pd.read_csv(os.path.join(data_path, 'fp_samples.csv'))
    Feature_len = floor_df.shape[1] - 6

    locs_null_index = list(floor_df.loc[floor_df['wp_ts'].isnull(), :].index)
    path_groups = floor_df.groupby(['path'])
    group_list = []
    seq_len = 5
    for name, group in path_groups:
        group_df = pd.DataFrame(group)
        if len(group_df)>=seq_len:
            group_df = group_df.sort_values(by='ts')
            ip_x_y = interpolate_rp(group_df)
            group_list.append(ip_x_y)
    floor_df = pd.concat(group_list, axis=0)
    floor_df_cp = floor_df.copy()
    floor_df_inter_index = floor_df_cp.index
    intersected_null_index = [i for i in floor_df_inter_index if i in locs_null_index]
    floor_df_cp.loc[intersected_null_index, ['x', 'y']] = np.nan  #

    # TMR groundtruth
    selected_floor_df = floor_df_cp.loc[(floor_df_cp['x']>=x_min)&(floor_df_cp['x']<=x_max)&(floor_df_cp['y']>=y_min)&(floor_df_cp['y']<=y_max),:]
    selected_index = selected_floor_df.index
    null_floor_df = pd.DataFrame(pd.isnull(selected_floor_df).sum())
    # print('null_floor_df', null_floor_df)
    selected_aps = null_floor_df[null_floor_df[0]==selected_floor_df.shape[0]].index
    floor_df_cp.loc[selected_index, selected_aps] = 12345.0
    number_index,number_cols = np.where(floor_df_cp.values==12345.0)
    number_of_TMRs= len(number_index)
    locs = floor_df.loc[:, ['x', 'y']]
    min_x_y = locs.loc[:, ['x', 'y']].min().values
    max_x_y = locs.loc[:, ['x', 'y']].max().values
    locs_values = (locs.loc[:, ['x', 'y']].values - min_x_y) / (max_x_y - min_x_y)

    fp_df = floor_df.iloc[:, :Feature_len]
    print('len of fp_df', len(fp_df))
    fp_masks = (~np.isnan(fp_df.values)).astype(int)

    # vary data quality
    row_index_v, col_index_v = np.where(fp_masks > 0)
    row_index_index_v = random.sample(range(len(row_index_v)), int(p_eta*0.1 * len(row_index_v)))
    test_row_indexs_v = row_index_v[row_index_index_v]
    test_col_indexs_v = col_index_v[row_index_index_v]
    print('len of test_row_indexs', len(test_row_indexs_v), len(test_col_indexs_v))
    fp_masks[test_row_indexs_v, test_col_indexs_v] = 0

    # find FMR truth
    row_index, col_index = np.where(fp_masks > 0)
    # row_index_index = random.sample(range(len(row_index)), int(0.01 * len(row_index)))
    times_of_TMRs = int(1/prop)
    row_index_index = random.sample(range(len(row_index)), number_of_TMRs*times_of_TMRs)

    print('FMR groudtruth samples', len(row_index_index))
    test_row_indexs = row_index[row_index_index]
    test_col_indexs = col_index[row_index_index]
    print('len of test_row_indexs', len(test_row_indexs), len(test_col_indexs))
    fp_masks[test_row_indexs, test_col_indexs] = 0

    original_index = fp_df.index
    fp_df = pd.DataFrame(fp_masks, index=original_index)

    km_input = np.concatenate([fp_masks, locs_values], axis=1)

    if method.startswith('tac'):
        km_input = np.concatenate([km_input, locs.values], axis=1)

    n_clusters = cluster

    for k in range(n_clusters, n_clusters+1):
        if method == 'akm':
            km = KMeans(n_clusters=k, random_state=2021)
            km.fit(km_input)
            labels = km.labels_
        elif method == 'akm':
            cus_hc = Hierarchical()
            cus_hc.fit(km_input)
            labels = cus_hc.labels
            k = len(list(set(labels)))
        fp_df['cluster_labels'] = labels
        fp_df = pd.concat([fp_df, locs],axis=1)
        fp_reconstrcuted_list = []
        recontructed_index_list = []
        for i in list(range(k)):
            clu_index = fp_df.loc[fp_df['cluster_labels']==i,:].index
            recontructed_index_list.extend(list(clu_index))
            fp_clu = fp_df.loc[fp_df['cluster_labels']==i,:].drop(['cluster_labels','x','y'], axis=1).values
            print('fp_clu', fp_clu.shape)
            fp_clu_sum = np.sum(fp_clu, axis=0)
            fp_clu_cols = np.where(fp_clu_sum>0)[0]

            fp_clu_cols_max = np.where((fp_clu_sum>threshold*fp_clu.shape[0])&(fp_clu_sum>1))[0]
            print(fp_clu_sum[fp_clu_cols])
            print(fp_clu_sum[fp_clu_cols_max])
            fp_clu_recontructed = copy.deepcopy(fp_clu)
            fp_clu_recontructed[:,fp_clu_cols_max] = 1
            fp_reconstrcuted_list.append(fp_clu_recontructed)
        fp_reconstrcuted = np.concatenate(fp_reconstrcuted_list, axis=0)
        fp_re_df = pd.DataFrame(fp_reconstrcuted, index=recontructed_index_list)
        fp_re_df = fp_re_df.loc[original_index,:]
        fp_re_values = fp_re_df.values
        recontruted_test_values = fp_re_values[test_row_indexs, test_col_indexs]
        print('P number', len(recontruted_test_values))
        print('TP number', sum(recontruted_test_values))
        print('re TPR %', round(sum(recontruted_test_values)/len(recontruted_test_values), 3))
        tpr = round(sum(recontruted_test_values)/len(recontruted_test_values), 3)

        recons_test_v = fp_re_values[number_index, number_cols]
        print('N number', len(recons_test_v))
        print('TN number', len(recons_test_v)-sum(recons_test_v))
        print('re TNR %', round(1-sum(recons_test_v)/len(recons_test_v), 3))
        tnr = round(1-sum(recons_test_v)/len(recons_test_v), 3)
        accuracy = (sum(recontruted_test_values)+(len(recons_test_v)-sum(recons_test_v)))/(len(recontruted_test_values)+len(recons_test_v))
        print('accuracy:',  round(accuracy, 3))
        accuracy = round(accuracy, 3)
        balanced_accuracy = 0.5*(sum(recontruted_test_values)/len(recontruted_test_values) + (1-sum(recons_test_v)/len(recons_test_v)))
        balanced_accuracy = round(balanced_accuracy, 3)
        print('balanced accuracy:', round(balanced_accuracy, 3))
        print('method:', method)
        print('clusters:',k)
        fpr = sum(recons_test_v)/len(recons_test_v)
        print('threshold:', threshold)
        print('tpr, fpr:', tpr, fpr)
        print('p_eta:', p_eta)
        return tpr, tnr, accuracy, balanced_accuracy

tprs = []
fprs = []
threld = 0.1
tpr_list = []
tnr_list = []
accuracy_list = []
balanced_accuracy_list = []
proportion_list = range(1,21,1)
for cluster in range(1,200, 1):  
    avg_ba_list = []
    for proportion in proportion_list:
        tpr, tnr, accuracy, balanced_accuracy = calculate_fpr_tpr(threshold=threld, p_eta=0.0, cluster=cluster, prop=proportion)
        tpr_list.append(tpr)
        tnr_list.append(tnr)
        accuracy_list.append(accuracy)
        balanced_accuracy_list.append(balanced_accuracy)
    avg_ba_list.append(np.average(balanced_accuracy_list))
clu_range = range(1,200, 1)
n_ba = list(zip(clu_range, balanced_accuracy_list))
best_n = sorted(n_ba, key=lambda x:x[1], reverse=True)[0]
print(best_n)
