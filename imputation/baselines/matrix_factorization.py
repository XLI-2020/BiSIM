
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


site = 'KDM'
data_root_path = '../data'
data_path = os.path.join(data_root_path, site)
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_samples'), header=0)
method = 'thac'
thre = 0.1
wifi_df = pd.read_csv(os.path.join(data_path, 'fp_filterd_{site}_{floor}_{method}_{thre}.csv'.format(site=site, method=method, thre=str(thre))), header=0)

testing_data_path = os.path.join(data_path, 'testing_data')
num_of_test_samples = 28
non_null_wifi_df = wifi_df.loc[~wifi_df['x'].isnull(), :].sample(frac=1).iloc[:num_of_test_samples, :]
test_index = non_null_wifi_df.index

testing_data_path = os.path.join(data_path, 'testing_data')
with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'r+') as file:
    test_all_index = json.load(file)

FEATURE_LEN = wifi_df.shape[1] - 6
test_index = test_index[:28]
test_df_xy = wifi_df.loc[test_index, ['x', 'y']]

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

n = len(X)
batch_size = 100
nb_batch = (n + batch_size - 1) // batch_size

# feat_bs = 128
nb_ft_bs = (len(X_cols) + batch_size - 1) // batch_size

print('nb_ft_bs', nb_ft_bs)
print('nb_batch', nb_batch)
mf = MatrixFactorization()

X = X.values
X_mf = []
for i in range(nb_batch):
    print('i',i)
    x = X[i * batch_size: (i + 1) * batch_size, :]
    x_mf_col = mf.fit_transform(x)
    X_mf.append(x_mf_col)

X_mf = np.concatenate(X_mf, axis=0)
X_re_df = pd.DataFrame(X_mf, index=X_index, columns=X_cols)
print('X_mf shape', X_mf.shape)
et = time.time()
print('after mf')
print(pd.isnull(X_re_df))
print(pd.isnull(X_re_df).sum())
print('elapsed time:', (et-st)/60)
# X_completed = X_re_df.fillna(0)

train_df = X_re_df.loc[train_index, :]
test_df = X_re_df.loc[test_index, :]
print('test_df', test_df.shape)
### evaluation
X_train = train_df.iloc[:,:FEATURE_LEN].astype(float).values
X_train = X_train * std + mean
# X_train = train_df.iloc[:,:FEATURE_LEN].apply(lambda x:pow(10, x/10)).astype(float).values
Y_train = train_df.loc[:, ['x', 'y']].astype(float).values
Y_train = Y_train * std_x_y + mean_x_y

X_test = test_df.iloc[:,:FEATURE_LEN].astype(float).values
X_test = X_test * std + mean


Y_test = test_df_xy.loc[:, ['x', 'y']].astype(float).values
print('Y_test', Y_test.shape)
# Y_test = Y_test * std_x_y + mean_x_y


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










