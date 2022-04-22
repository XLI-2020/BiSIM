import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
import pandas as pd
import json
import os
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
import math
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
random.seed(2020)
def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compare_dist_between_two_points_list(pl1, pl2):
    sum_dist = sum(list(map(lambda x: dist(pl1[x], pl2[x]), range(len(pl1)))))
    avg_dist = sum_dist / float(len(pl1))
    return avg_dist

def get_localization_dist(radio_map, reference_points, fingprint_sample, ground_points):
    X_train = np.array(radio_map)
    Y_train = np.array(reference_points)
    X_test = np.array(fingprint_sample)
    Y_test = np.array(ground_points)
    knn = KNeighborsRegressor(n_neighbors=3)
    wknn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(X_train, Y_train)
    wknn.fit(X_train, Y_train)
    Y_pre = knn.predict(X_test)
    w_Y_pre = wknn.predict(X_test)
    avg_dist = compare_dist_between_two_points_list(Y_pre, Y_test)
    w_avg_dist = compare_dist_between_two_points_list(w_Y_pre, Y_test)
    avg_dist = round(avg_dist, 6)
    w_avg_dist = round(w_avg_dist, 6)
    return avg_dist, w_avg_dist

def get_mean_var_db(method, site, floor, thre):
    site = site
    floor_num = floor
    data_root_path = '../data'
    data_path = os.path.join(data_root_path, site)
    wifi_df = pd.read_csv(os.path.join(data_path, 'fp_filterd_{site}_{method}_{thre}.csv'.format(site=site, method=method, thre=thre)), header=0)
    all_null_df = pd.DataFrame(pd.isnull(wifi_df[wifi_df == -100]).sum(axis=0))
    all_null_cols = list(all_null_df[all_null_df[0] == 0].index)
    wifi_df = wifi_df.drop(all_null_cols, axis=1)
    other_columns = ['floor', 'x', 'y', 'wp_ts', 'ts', 'path']
    mean = wifi_df.drop(other_columns, axis=1).mean().values
    std = wifi_df.drop(other_columns, axis=1).std().values
    mean_x_y = wifi_df.loc[:, ['x', 'y']].mean().values
    std_x_y = wifi_df.loc[:, ['x', 'y']].std().values
    return mean, std, mean_x_y, std_x_y



