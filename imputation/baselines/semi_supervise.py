import copy
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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def knn_position(X_train, Y_train, X_test, Y_test):
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
    return avg_dist

site = 'KDM'
data_root_path = '../data'
data_path = os.path.join(data_root_path, site)
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_samples'), header=0)
method = 'thac'
thre = 0.1
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_filterd_{site}_{floor}_{method}_{thre}.csv'.format(site=site, method=method, thre=str(thre))), header=0)

testing_data_path = os.path.join(data_path, 'testing_data')
with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'r+') as file:
    test_all_index = json.load(file)

FEATURE_LEN = wifi_df.shape[1] - 6
feat_df = wifi_df.iloc[:,:FEATURE_LEN]
feat_df = feat_df.fillna(-100)
wifi_df = pd.concat([feat_df, wifi_df.iloc[:, FEATURE_LEN:]], axis=1)
test_index = test_all_index[:28]
val_index = test_all_index[28:60]
val_df = wifi_df.loc[val_index, :]

val_x = val_df.iloc[:,:FEATURE_LEN].values
val_y = val_df.loc[:, ['x','y']].values

test_df = wifi_df.loc[test_index,:]
print('test_index', len(test_index))
print('len of wifi df', len(wifi_df))

test_x = test_df.iloc[:,:FEATURE_LEN].values
test_y = test_df.loc[:, ['x','y']].values


train_data = wifi_df.drop(test_all_index, axis=0)
print('wifi_df after drop test index', wifi_df.shape)
print('non-null length before interpolation', len(wifi_df.loc[~wifi_df['wp_ts'].isnull(),:]))
comp_train_data = train_data.loc[~train_data['wp_ts'].isnull(),:]
incomp_train_data = train_data.loc[train_data['wp_ts'].isnull(),:].sample(frac=1)

n = len(incomp_train_data)
print('n', n)
batch_size = 100
nb_batch = (n + batch_size - 1) // batch_size

S_X_train = comp_train_data.iloc[:,:FEATURE_LEN].values
S_Y_train = comp_train_data.loc[:, ['x', 'y']].values
# svr = SVR()
rf = RandomForestRegressor()
batch_list = list(range(nb_batch))
iter = 0

while batch_list:
    pos_dist_list = []
    temp_train_list = []
    print('iter', iter)
    estimator = copy.copy(rf)
    estimator.fit(S_X_train, S_Y_train)
    for index, j in enumerate(batch_list):
        print('j:,{j}'.format(j=j))
        x = incomp_train_data.iloc[j * batch_size: (j + 1) * batch_size,:]
        x_feat = x.iloc[:, :FEATURE_LEN].values
        y_pred = estimator.predict(x_feat)
        temp_X_train = np.concatenate([S_X_train, x_feat], axis=0)
        temp_Y_train = np.concatenate([S_Y_train, y_pred], axis=0)
        temp_train_list.append((index,(temp_X_train, temp_Y_train)))
        pos_dist = knn_position(temp_X_train, temp_Y_train, val_x, val_y)
        pos_dist_list.append((index, pos_dist))
    best_batch_index, best_pos_res = list(sorted(pos_dist_list, key=lambda x:x[1], reverse=False))[0]
    print('best_pos_res:', best_pos_res)

    S_X_train, S_Y_train = list(filter(lambda x:x[0] == best_batch_index, temp_train_list))[0][1]
    del batch_list[best_batch_index]
    iter += 1
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
    Y_pre = get_pos_results(S_X_train, S_Y_train, test_x,estimator_name=model)
    avg_dist = compare_dist_between_two_points_list(Y_pre, test_y)
    print(avg_dist)
    pos_results[model] = avg_dist

print('pos_results', pos_results)


