import json
import os
import time

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import sys
import random

def find_waypoint(t, wp_df):
    wp_df = wp_df.sort_values(by=['wp_ts'], ascending=True)
    wp_df_a = wp_df.shift(-1).rename(columns={'wp_ts':'a_ts', 'x':'a_x', 'y':'a_y'})
    wp_df_c = pd.concat([wp_df, wp_df_a], axis=1)
    target_row = wp_df_c.loc[(wp_df_c['wp_ts']<=t)&(wp_df_c['a_ts']>t),['wp_ts','x','y','a_ts', 'a_x', 'a_y']].reset_index(drop=True)
    if not target_row.empty:
        x = target_row.loc[0,'x'] + ((target_row.loc[0,'a_x']-target_row.loc[0,'x'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        y = target_row.loc[0,'y'] + ((target_row.loc[0,'a_y']-target_row.loc[0,'y'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        return x, y
    else:
        return None, None
site = 'site3'
floor_num = 'F1'

derived_data_path = '../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)

wifi_df = pd.read_csv(os.path.join(derived_data_path, 'fp_filterd_{}.csv'.format(floor_num)), header=0)
# testing_data_path = os.path.join(derived_data_path, 'testing_data')
# with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'r+') as file:
#     test_index = json.load(file)

brits_data_path = os.path.join(derived_data_path, 'biseq')
with open(os.path.join(brits_data_path, 'val_path_names.json'), 'r+') as file:
    path_names = json.load(file)

test_index = wifi_df.loc[(wifi_df['path'].isin(path_names))&(~wifi_df['wp_ts'].isnull()),:].index

FEATURE_LEN = 1983

test_df = wifi_df.loc[test_index,:]

wifi_df = wifi_df.drop(test_index, axis=0)

print('wifi_df after drop test index', wifi_df.shape)
print('non-null length before interpolation', len(wifi_df.loc[~wifi_df['wp_ts'].isnull(),:]))
### interpolation
path_groups = wifi_df.groupby(['path'])
print('len of paths', len(path_groups))
lp_list = []
st = time.time()

# [6400, 10130, 5706, 1638]
for path, group in path_groups:
    group_df = pd.DataFrame(group)
    have_wp_df = group_df.loc[~group_df['wp_ts'].isnull(),:]
    print('have_wp_df shape', have_wp_df.shape)
    wp_df = have_wp_df.loc[:, ['wp_ts', 'x', 'y']]
    have_null_wp_df = group_df.loc[group_df['wp_ts'].isnull(),:]
    print('have_null_wp_df shape', have_null_wp_df.shape)
    if not have_null_wp_df.empty:
        for index, row in have_null_wp_df.iterrows():
            ts = row['ts']
            x, y = find_waypoint(ts, wp_df)
            if x and y:
                have_null_wp_df.loc[index,'x'] = x
                have_null_wp_df.loc[index,'y'] = y
                have_null_wp_df.loc[index,'wp_ts'] = ts
    # have_null_wp_df_after_inter = have_null_wp_df.loc[have_null_wp_df['wp_ts'].isnull(),:]
    # print('have_null_wp_df_after_inter shape', have_null_wp_df_after_inter.shape)
    group_df = pd.concat([have_wp_df, have_null_wp_df],axis=0)
    lp_list.append(group_df)

wifi_df_interpolated = pd.concat(lp_list, axis=0)

print('non-null length after interpolation', len(wifi_df_interpolated.loc[~wifi_df_interpolated['wp_ts'].isnull(),:]))

wifi_df_in = wifi_df_interpolated.loc[~wifi_df_interpolated['wp_ts'].isnull(),:]
print('wifi df after interpolation', wifi_df_in.shape)


# validation_data_path = os.path.join(derived_data_path, 'validation_data')
# with open(os.path.join(validation_data_path, 'validate_data_index_2021.json'), 'r+') as file:
#     validation_index = json.load(file)
# testing_data_path = os.path.join(derived_data_path, 'testing_data')
# with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'r+') as file:
#     test_index = json.load(file)

# train_index = list(set(list(wifi_df_in.index)) - set(validation_index) - set(test_index))
train_index = list(set(list(wifi_df_in.index)))

train_df = wifi_df_in.loc[train_index,:]

# print(pd.isnull(train_df))
# test_df = wifi_df_in.loc[test_index,:]

test_df = test_df.fillna(-100)
train_df = train_df.fillna(-100)



### evaluation
X_train = train_df.iloc[:,:FEATURE_LEN].astype(float).values

# X_train = train_df.iloc[:,:FEATURE_LEN].apply(lambda x:pow(10, x/10)).astype(float).values

Y_train = train_df.loc[:, ['x', 'y']].astype(float).values

X_test = test_df.iloc[:,:FEATURE_LEN].astype(float).values
# X_test = test_df.iloc[:,:FEATURE_LEN].apply(lambda x:pow(10, x/10)).astype(float).values

Y_test = test_df.loc[:, ['x', 'y']].astype(float).values

knn = KNeighborsRegressor(n_neighbors=3,weights='distance')
# knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(X_train, Y_train)
Y_pre = knn.predict(X_test)

print(Y_pre[:3])
print(Y_test[:3])

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def compare_dist_between_two_points_list(pl1, pl2):
    sum_dist = sum(list(map(lambda x:dist(pl1[x], pl2[x]),range(len(pl1)))))
    avg_dist = sum_dist/float(len(pl1))
    return avg_dist


avg_dist = compare_dist_between_two_points_list(Y_pre, Y_test)
print(avg_dist)



