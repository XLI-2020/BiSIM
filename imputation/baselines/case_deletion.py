import json
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import sys
import random
site = 'KDM'
data_root_path = '../data'
data_path = os.path.join(data_root_path, site)
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_samples'), header=0)

wifi_df = wifi_df.loc[~wifi_df['wp_ts'].isnull(),:]
FEATURE_LEN = wifi_df.shape[1] - 6
wifi_df = wifi_df.sample(frac=1, random_state=2021)
wifi_df = wifi_df.fillna(-100)
train_df = wifi_df.head(int(len(wifi_df)*0.8))
test_df = wifi_df.tail(int(len(wifi_df)*0.2))

print('test_df shape', test_df.shape)
test_index = list(test_df.index)

testing_data_path = os.path.join(data_path, 'testing_data')
if not os.path.exists(testing_data_path):
    os.mkdir(testing_data_path)
with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'w+') as file:
    json.dump(test_index, file)

X_train = train_df.iloc[:,:FEATURE_LEN].astype(float).values
Y_train = train_df.loc[:, ['x', 'y']].astype(float).values
X_test = test_df.iloc[:,:FEATURE_LEN].astype(float).values
Y_test = test_df.loc[:, ['x', 'y']].astype(float).values

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
