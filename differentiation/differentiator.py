import os
import random
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import copy
from hc_custom_utils import Hierarchical, interpolate_rp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--site', type = str, default='KDM')
parser.add_argument('--method', type = str, default='akm')
parser.add_argument('--thre', type = float, default=0.1)
args = parser.parse_args()

random.seed(2021)
data_root_path = '../data'
site = args.site
method = args.method
thre = args.thre

# load data
data_root_path = '../imputation/data'
data_path = os.path.join(data_root_path, site)
floor_df = pd.read_csv(os.path.join(data_path, 'fp_samples.csv'))
Feature_len = floor_df.shape[1]-6
locs_null_index = list(floor_df.loc[floor_df['wp_ts'].isnull(), :].index)
path_groups = floor_df.groupby(['path'])
group_list = []
seq_len = 5
# slice long sequences
for name, group in path_groups:
    group_df = pd.DataFrame(group)
    if len(group_df) >= seq_len:
        group_df = group_df.sort_values(by='ts')
        ip_x_y = interpolate_rp(group_df)
        group_list.append(ip_x_y)
floor_df = pd.concat(group_list, axis=0)
floor_df_cp = floor_df.copy()

floor_df_inter_index = floor_df_cp.index
intersected_null_index = [i for i in floor_df_inter_index if i in locs_null_index]
floor_df_cp.loc[intersected_null_index, ['x', 'y']] = np.nan  #

locs = floor_df.loc[:,['x', 'y']]
min_x_y = locs.loc[:,['x','y']].min().values
max_x_y = locs.loc[:,['x','y']].max().values
locs_values = (locs.loc[:,['x','y']].values - min_x_y)/(max_x_y - min_x_y)
fp_df = floor_df.iloc[:,:Feature_len]
side_info_cols = ['floor', 'x', 'y', 'wp_ts','ts', 'path']
side_df = floor_df_cp.loc[:, side_info_cols]

fp_concre_df = fp_df.copy()
fp_masks = (~np.isnan(fp_df.values)).astype(int)
original_index = fp_df.index
fp_df = pd.DataFrame(fp_masks, index=original_index)

#concatenation of fingerprint and normalized locations
km_input = np.concatenate([fp_masks, locs_values], axis=1)
if method.startswith('tac'):
    km_input = np.concatenate([km_input, locs.values], axis=1)

n_clusters = 50
color = cm.rainbow(np.linspace(0, 1, n_clusters))
def generate_detected_results(thre):
    #clustering based
    for k in range(n_clusters, n_clusters+1):
        if method == 'akm':
            km = KMeans(n_clusters=k, random_state=2021)
            km.fit(km_input)
            labels = km.labels_
        elif method == 'tac':
            cus_hc = Hierarchical()
            cus_hc.fit(km_input)
            labels = cus_hc.labels
            k = len(list(set(labels)))
            print('new k', k)
        fp_df['cluster_labels'] = labels
        fp_reconstrcuted_list = []
        recontructed_index_list = []
        # within clusering differentiation
        for i in list(range(k)):
            clu_index = fp_df.loc[fp_df['cluster_labels'] == i, :].index
            recontructed_index_list.extend(list(clu_index))
            fp_clu = fp_df.loc[fp_df['cluster_labels'] == i, :].drop(['cluster_labels'], axis=1).values
            fp_clu_sum = np.sum(fp_clu, axis=0)
            fp_clu_recontructed = copy.deepcopy(fp_clu)
            threshold = thre
            fp_clu_cols_max = np.where((fp_clu_sum > threshold * fp_clu.shape[0]))[0]
            fp_clu_recontructed[:, fp_clu_cols_max] = 1
            fp_reconstrcuted_list.append(fp_clu_recontructed)
        fp_reconstrcuted = np.concatenate(fp_reconstrcuted_list, axis=0)
        fp_re_df = pd.DataFrame(fp_reconstrcuted, index=recontructed_index_list)
        fp_re_df = fp_re_df.loc[original_index,:]
        fp_re_values = fp_re_df.values
        fp_concre_df.values[np.where(fp_re_values == 0)] = -100
        fp_final_df = pd.concat([fp_concre_df, side_df], axis=1)
        fp_final_df.to_csv(os.path.join(data_path, 'fp_filterd_{site}_{method}_{thre}.csv'.format(site=site,  method=method, thre=str(thre))), index=False,
                           header=True)
generate_detected_results(thre)






