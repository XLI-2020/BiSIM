import os
import random
import time
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from hc_custom_utils import Hierarchical, interpolate_rp

random.seed(2021)
data_root_path = '../data'
site = 'KDM'
method = 'elbow'
p_eta = 0.1
data_root_path = '../imputation/data'
data_path = os.path.join(data_root_path, site)
floor_df = pd.read_csv(os.path.join(data_path, 'fp_samples.csv'))
Feature_len = floor_df.shape[1] - 6

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

locs = floor_df.loc[:,['x', 'y']]
min_x_y = locs.loc[:,['x','y']].min().values
max_x_y = locs.loc[:,['x','y']].max().values
locs_values = (locs.loc[:,['x','y']].values - min_x_y)/(max_x_y - min_x_y)
fp_df = floor_df.iloc[:,:Feature_len]
side_info_cols = ['floor', 'x', 'y', 'wp_ts','ts', 'path']
side_df = floor_df_cp.loc[:, side_info_cols]

fp_concre_df = fp_df.copy()
fp_masks = (~np.isnan(fp_df.values)).astype(int)

#vary data quality
row_index, col_index = np.where(fp_masks > 0)
row_index_index = random.sample(range(len(row_index)), int(p_eta*0.1 * len(row_index)))
test_row_indexs = row_index[row_index_index]
test_col_indexs = col_index[row_index_index]
print('len of test_row_indexs', len(test_row_indexs), len(test_col_indexs))
fp_masks[test_row_indexs, test_col_indexs] = 0
fp_concre_df.iloc[test_row_indexs, test_col_indexs] = np.nan

original_index = fp_df.index
fp_df = pd.DataFrame(fp_masks, index=original_index)

km_input = np.concatenate([fp_masks, locs_values], axis=1)
if method.startswith('tac'):
    km_input = np.concatenate([km_input, locs.values], axis=1)

n_clusters = 10
n_clu_upper = 200
clu_range_interval = 10
color = cm.rainbow(np.linspace(0, 1, n_clusters))
inertia_list = []
for k in range(n_clusters, n_clu_upper, clu_range_interval):
    if method == 'elbow':
        km = KMeans(n_clusters=k, random_state=2021)
        km.fit(km_input)
        labels = km.labels_
        inertia = km.inertia_
        inertia_list.append(inertia)
    elif method == 'tac':
        cus_hc = Hierarchical()
        cus_hc.fit(km_input)
        labels = cus_hc.labels
        k = len(list(set(labels)))

clu_range = range(n_clusters, n_clu_upper, clu_range_interval)
k_inertia = list(zip(clu_range, inertia_list))
print('k_inertia', k_inertia)

plt.plot(clu_range, inertia_list)
plt.xticks(clu_range)
plt.show()




