import fancyimpute
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
from fancyimpute import NuclearNormMinimization, MatrixFactorization, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


site = 'site5'
floor_num = 'F4'
method = 'thac'
thre = 0.1
derived_data_path = '../../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)
wifi_df = pd.read_csv(os.path.join(derived_data_path, 'fp_filterd_{site}_{floor}_{method}_{thre}.csv'.format(site=site, floor=str(floor_num), method=method, thre=str(thre))), header=0)
# wifi_df = pd.read_csv(os.path.join(derived_data_path, 'fp_sample_{floor}.csv'.format(floor=str(floor_num))), header=0)


print('wifi_df', wifi_df.shape)
num_of_test_samples = 28
non_null_wifi_df = wifi_df.loc[~wifi_df['x'].isnull(), :].sample(frac=1).iloc[:num_of_test_samples, :]
print('non_null_wifi_df', non_null_wifi_df.shape)
test_index = non_null_wifi_df.index

print('test data samples:', wifi_df.loc[test_index[:3], ['x', 'y']])

# brits_data_path = os.path.join(derived_data_path, 'biseq')
# with open(os.path.join(brits_data_path, 'val_path_names.json'), 'r+') as file:
#     path_names = json.load(file)
# test_index = wifi_df.loc[(wifi_df['path'].isin(path_names))&(~wifi_df['wp_ts'].isnull()),:].index
FEATURE_LEN = wifi_df.shape[1] - 6
test_index = test_index[:]
test_df_xy = wifi_df.loc[test_index, ['x', 'y']]

print('test_df_xy', test_df_xy.loc[test_index[:3],:])
train_index = wifi_df.drop(test_index, axis=0).index

feature_df = wifi_df.iloc[:,:FEATURE_LEN]
xy_df = wifi_df.loc[:,['x', 'y']]
xy_df.loc[test_index, ['x', 'y']] = np.nan

other_columns = ['floor', 'x', 'y', 'wp_ts','ts', 'path']
mean = wifi_df.drop(other_columns, axis=1).mean().values
std = wifi_df.drop(other_columns, axis=1).std().values
mean_x_y = wifi_df.loc[:,['x','y']].mean().values
std_x_y = wifi_df.loc[:,['x','y']].std().values

feature_df = (feature_df - mean)/std
xy_df = (xy_df - mean_x_y)/std_x_y

X = pd.concat([feature_df, xy_df], axis=1)

X_index = X.index
X_cols = X.columns

print('X',X.shape)


st = time.time()
print('before mf')
print(pd.isnull(X).sum())
mice = IterativeImputer()

n = len(X)
batch_size = 300

nb_ft_bs = (len(X_cols) + batch_size - 1) // batch_size
X = X.values
X_mice = []

x_mice_col_list = []
for j in range(nb_ft_bs):
    print('j:,{j}'.format(j=j))
    x = X[:, j * batch_size: (j + 1) * batch_size]
    print('x shaope', x.shape)
    x_mice_col = mice.fit_transform(x)
    print('x_mice_col shape', x_mice_col.shape)
    x_mice_col_list.append(x_mice_col)

X_mice_col = np.concatenate(x_mice_col_list, axis=1)
print('X_mice_col shape', X_mice_col.shape)

X_mice = X_mice_col.copy()

# X_mice = np.concatenate(X_mice, axis=0)
X_re_df = pd.DataFrame(X_mice, index=X_index, columns=X_cols)

print('X_mice shape', X_mice.shape)
et = time.time()
print('after mice')
print(pd.isnull(X_re_df))
print(pd.isnull(X_re_df).sum())
print('elapsed time:', (et-st)/60)

X_completed = X_re_df.fillna(0)

train_df = X_completed.loc[train_index, :]
test_df = X_completed.loc[test_index, :]
### evaluation
X_train = train_df.iloc[:,:FEATURE_LEN].astype(float).values
X_train = X_train * std + mean


Y_train = train_df.loc[:, ['x', 'y']].astype(float).values
Y_train = Y_train * std_x_y + mean_x_y


X_test = test_df.iloc[:,:FEATURE_LEN].astype(float).values
X_test = X_test * std + mean

Y_test = test_df_xy.loc[:, ['x', 'y']].astype(float).values

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def compare_dist_between_two_points_list(pl1, pl2):
    sum_dist = sum(list(map(lambda x:dist(pl1[x], pl2[x]),range(len(pl1)))))
    avg_dist = sum_dist/float(len(pl1))
    return avg_dist

def get_pos_results(X_train, Y_train, X_test, estimator_name=None):
    if estimator_name == 'knn':
        estimator = KNeighborsRegressor(n_neighbors=3)
    elif estimator_name == 'w-knn':
        estimator = KNeighborsRegressor(n_neighbors=3, weights='distance')
    elif estimator_name == 'rf':
        estimator = RandomForestRegressor()
    elif estimator_name == 'svr':
        estimator_x = SVR()
        estimator_y = SVR()

    if estimator_name in ['svr']:
        estimator_x.fit(X_train, Y_train[:, 0].tolist())
        estimator_y.fit(X_train, Y_train[:, 1].tolist())
        Y_pred_x = estimator_x.predict(X_test).reshape(-1,1)
        Y_pred_y = estimator_y.predict(X_test).reshape(-1,1)
        Y_pred = np.concatenate([Y_pred_x, Y_pred_y], axis=1)
        print('Y_pred shape', Y_pred.shape)
    else:
        estimator.fit(X_train, Y_train)
        Y_pred = estimator.predict(X_test)

    return Y_pred

pos_results = {}
for model in ['knn', 'w-knn', 'svr', 'rf']:
    Y_pre = get_pos_results(X_train, Y_train, X_test, estimator_name=model)
    avg_dist = compare_dist_between_two_points_list(Y_pre, Y_test)
    print(avg_dist)
    pos_results[model] = avg_dist

print('pos_results', pos_results)














