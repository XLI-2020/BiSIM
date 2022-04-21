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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.cm as cm
from hc_custom_utils import Hierarchical, interpolate_rp
from hc_custom_utils_dynamic import Hierarchical as hc_d

random.seed(2021)
data_root_path = '../data'
site = 'site4'
floor_num = 'F1'
method = 'kmeans'
cluster_type = '1'
p_eta = 0.3
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
row_index_index = random.sample(range(len(row_index)), int(p_eta*0.1 * len(row_index)))
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


n_clusters = 10
n_clu_upper = 100
clu_range_interval = 10
color = cm.rainbow(np.linspace(0, 1, n_clusters))
# geometry = geo_referencing()
# plot_shape(geometry)
inertia_list = []
for k in range(n_clusters, n_clu_upper, clu_range_interval):
    if method == 'kmeans':
        km = KMeans(n_clusters=k, random_state=2021)
        km.fit(km_input)
        labels = km.labels_
        inertia = km.inertia_
        inertia_list.append(inertia)
        print('kmeans k', k)
    elif method == 'agg':
        agghc = AgglomerativeClustering(n_clusters=k, linkage='complete')
        agghc.fit(km_input)
        labels = agghc.labels_
    elif method == 'cust':
        cus_hc = Hierarchical()
        cus_hc.fit(km_input)
        labels = cus_hc.labels
        k = len(list(set(labels)))
        print('new k', k)
    elif method == 'cust_d':
        cus_hc = hc_d()
        cus_hc.fit(km_input)
        labels = cus_hc.labels
        k = len(list(set(labels)))
        print('new k', k)

clu_range = range(n_clusters, n_clu_upper, clu_range_interval)
k_inertia = list(zip(clu_range, inertia_list))
print('k_inertia', k_inertia)

plt.plot(clu_range, inertia_list)
plt.xticks(clu_range)
plt.show()

"""

site5:
eta=0.1 n=30
eta=0.2 n=30
eta=0.3 n=60


site4:
eta=0.1 n=20
eta=0.2 n=30
eta=0.3 n=30

"""


