import os
import random
import sys
import time

sys.path.append('/Users/xiaol/PycharmProjects/WifiLocalization')
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.cm as cm
import copy
from hc_custom_utils import Hierarchical, interpolate_rp
# from hc_custom_utils_dynamic import Hierarchical as hc_d
def generate_samples(eta):
    random.seed(2021)
    data_root_path = '../data'
    site = 'site4'
    floor_num = 'F1'
    method = 'cust'
    cluster_type = '1'
    p_eta = eta

    if site == 'site5':
        if method == 'elkm':
            if p_eta == 0.1:
                n_clusters = 30
            elif p_eta == 0.2:
                n_clusters = 30
            elif p_eta == 0.3:
                n_clusters = 60
        elif method == 'akm':
            if p_eta == 0.1:
                n_clusters = 70
            elif p_eta == 0.2:
                n_clusters = 90
            elif p_eta == 0.3:
                n_clusters = 80
    elif site == 'site4':
        if method == 'elkm':
            if p_eta == 0.01:
                n_clusters = 20
            elif p_eta == 0.02:
                n_clusters = 30
            elif p_eta == 0.03:
                n_clusters = 30
        elif method == 'akm':
            if p_eta == 0.01:
                n_clusters = 160
            elif p_eta == 0.02:
                n_clusters = 130
            elif p_eta == 0.03:
                n_clusters = 90
    data_path = os.path.join(data_root_path, site)
    print('site, floor:', site, floor_num)
    floor_dirs = Path(data_path).resolve().glob("./*/")
    floor_bssids = {}
    floor_dirs = list(filter(lambda x:str(x).split('/')[-1][0] != '.', floor_dirs))
    derived_data_path = '../derived_data/{site}'.format(site=site)
    with open(os.path.join(derived_data_path, 'wifi_bssids.json'), 'r+') as file:
        wifi_bssids = json.load(file)
    building_bssids = list(set(list(chain(*(wifi_bssids.values())))))
    print(len(building_bssids))
    floor_sample_path = '../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)
    floor_df = pd.read_csv(os.path.join(floor_sample_path, 'fp_sample_{}.csv'.format(str(floor_num))))
    print('before interpolation', floor_df.shape)
    Feature_len = floor_df.shape[1] - 6

    print('Feature_len', Feature_len)
    locs_null_index = list(floor_df.loc[floor_df['wp_ts'].isnull(), :].index)
    path_groups = floor_df.groupby(['path'])
    group_list = []
    seq_len = 5
    for name, group in path_groups:
        group_df = pd.DataFrame(group)
        if len(group_df) < seq_len:
            continue
        else:
            group_df = group_df.sort_values(by='ts')
            ip_x_y = interpolate_rp(group_df)
            group_list.append(ip_x_y)

    floor_df = pd.concat(group_list, axis=0)
    floor_df_cp = floor_df.copy()

    floor_df_inter_index = floor_df_cp.index
    intersected_null_index = [i for i in floor_df_inter_index if i in locs_null_index]
    floor_df_cp.loc[intersected_null_index, ['x', 'y']] = np.nan  #

    print('after interpolation', floor_df.shape)
    locs = floor_df.loc[:,['x', 'y']]
    min_x_y = locs.loc[:,['x','y']].min().values
    max_x_y = locs.loc[:,['x','y']].max().values
    locs_values = (locs.loc[:,['x','y']].values - min_x_y)/(max_x_y - min_x_y)
    fp_df = floor_df.iloc[:,:Feature_len]
    side_info_cols = ['floor', 'x', 'y', 'wp_ts','ts', 'path']
    side_df = floor_df_cp.loc[:, side_info_cols]

    fp_concre_df = fp_df.copy()
    print('len of fp_df', len(fp_df))
    fp_masks = (~np.isnan(fp_df.values)).astype(int)

    #vary data quality
    row_index, col_index = np.where(fp_masks > 0)
    row_index_index = random.sample(range(len(row_index)), int(p_eta* len(row_index)))
    test_row_indexs = row_index[row_index_index]
    test_col_indexs = col_index[row_index_index]
    print('len of test_row_indexs', len(test_row_indexs), len(test_col_indexs))
    fp_masks[test_row_indexs, test_col_indexs] = 0
    fp_concre_df.iloc[test_row_indexs, test_col_indexs] = np.nan

    original_index = fp_df.index
    fp_df = pd.DataFrame(fp_masks, index=original_index)
    # val_col_index = [i+1983 for i in val_col_index]

    if cluster_type == "combine":
        km_input = np.concatenate([fp_masks, locs_values], axis=1)
    else:
        km_input = fp_masks
    if method.startswith('cust'):
        km_input = np.concatenate([km_input, locs.values], axis=1)
    print('km_input', km_input.shape)

    # if site == 'site4' and thre == 0.2:
    #     n_clusters = 5
    # elif site == 'site4' and thre == 0.3:
    #     n_clusters = 10
    # elif site == 'site5' and thre == 0.2:
    #     n_clusters = 9
    # elif site == 'site5' and thre == 0.3:
    #     n_clusters = 13

    # color = cm.rainbow(np.linspace(0, 1, n_clusters))

    # geometry = geo_referencing()
    # plot_shape(geometry)
    def generate_detected_results(thre):
            if method in ['elkm', 'akm']:
                km = KMeans(n_clusters=n_clusters, random_state=2021)
                km.fit(km_input)
                labels = km.labels_
            elif method == 'agg':
                agghc = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
                agghc.fit(km_input)
                labels = agghc.labels_
            elif method == 'cust':
                cus_hc = Hierarchical()
                cus_hc.fit(km_input)
                labels = cus_hc.labels
                k = len(list(set(labels)))
                print('new k', k)
            elif method == 'dbscan':
                final_clusters = []
                # for eps in range(1, 1000, 10):
                #     for min_samples in range(1, 100,2):
                #         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                #         dbscan.fit(km_input)
                #         labels = dbscan.labels_
                #         k = len(list(set(labels)))
                #         final_clusters.append(k)

                dbscan = DBSCAN(eps=0.8, min_samples=1)
                dbscan.fit(km_input)
                labels = dbscan.labels_
                k = len(list(set(labels)))
                final_clusters.append(k)

                print('labels', labels)
                print('new k for dbscan', k)
                print('final clusters:', final_clusters)
                normal_k = list(filter(lambda x:x>1, final_clusters))
                print('normal k:', normal_k)
            elif method == 'knn':
                pass
            # elif method == 'cust_d':
            #     cus_hc = hc_d()
            #     cus_hc.fit(km_input)
            #     labels = cus_hc.labels
            #     k = len(list(set(labels)))
            #     print('new k', k)
            fp_df['cluster_labels'] = labels
            fp_reconstrcuted_list = []
            recontructed_index_list = []

            for i in list(range(k)):
                clu_index = fp_df.loc[fp_df['cluster_labels'] == i, :].index
                recontructed_index_list.extend(list(clu_index))
                fp_clu = fp_df.loc[fp_df['cluster_labels'] == i, :].drop(['cluster_labels'], axis=1).values
                print('fp_clu', fp_clu.shape)
                fp_clu_sum = np.sum(fp_clu, axis=0)
                fp_clu_recontructed = copy.deepcopy(fp_clu)
                fp_clu_cols = np.where(fp_clu_sum > 0)[0]
                threshold = thre
                fp_clu_cols_max = np.where((fp_clu_sum > threshold * fp_clu.shape[0]))[0]
                # fp_clu_recontructed = np.zeros(fp_clu.shape)
                fp_clu_recontructed[:, fp_clu_cols_max] = 1
                fp_reconstrcuted_list.append(fp_clu_recontructed)
            fp_reconstrcuted = np.concatenate(fp_reconstrcuted_list, axis=0)
            print('fp_reconstrcuted', fp_reconstrcuted.shape)
            fp_re_df = pd.DataFrame(fp_reconstrcuted, index=recontructed_index_list)
            fp_re_df = fp_re_df.loc[original_index,:]
            print('fp_re_df', fp_re_df.shape)
            fp_re_values = fp_re_df.values
            print('before fill', pd.isnull(fp_concre_df).sum(axis=0))
            fp_concre_df.values[np.where(fp_re_values == 0)] = -100
            print('after fill', pd.isnull(fp_concre_df).sum(axis=0))
            fp_final_df = pd.concat([fp_concre_df, side_df], axis=1)
            print('final', fp_final_df.shape)
            fp_final_df.to_csv(os.path.join(floor_sample_path, 'fp_filterd_{site}_{floor}_{method}_eta_{p_eta}.csv'.format(site=site, floor=str(floor_num), method=method, p_eta=str(p_eta))), index=False,
                               header=True)
    for thre in [0.1]:
        generate_detected_results(thre)

st = time.time()
for eta in [0.005, 0.0015]:
    generate_samples(eta)

et = time.time()

print('delta t:', (et-st)/60)
'''
site4 F1 akm:
thre=0.2  n=5
thre=0.3 n=10

site5 F4 akm: 
thre=0.2 n=9
thre=0.3  n=13
'''
'''
site5 F4 elkm: 
n=22
'''



