import json
import os

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import sys
import random
# np.random.seed(2021)
site = 'site3'
floor_num = 'F1'

derived_data_path = '../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)

# wifi_df = pd.read_csv(os.path.join(derived_data_path, 'fp_sample_{}.csv'.format(floor_num)), header=0)
wifi_df = pd.read_csv(os.path.join(derived_data_path, 'fp_filterd_{}.csv'.format(floor_num)), header=0)


# floor_int_dict = {'path_data_files':0, 'F1':1, 'F2':1, 'F3':2, 'F4':3, 'F5':4, 'F6':5, 'F7':6, 'F8':7}
# wifi_df['floor'] = wifi_df['floor'].map(floor_int_dict)

print(wifi_df.shape)
wifi_df = wifi_df.loc[~wifi_df['wp_ts'].isnull(),:]
print(wifi_df.shape)



# train_data_path = os.path.join(derived_data_path, 'training_data')
# with open(os.path.join(train_data_path, 'train_data_index_2021.json'), 'r+') as file:
#     train_index = json.load(file)
#
# test_df = wifi_df.loc[test_index, :]
# train_df = wifi_df.loc[train_index, :]
# test_df = test_df.fillna(-100)
# train_df = train_df.fillna(-100)


FEATURE_LEN = 1983
wifi_df = wifi_df.sample(frac=1, random_state=2021)
wifi_df = wifi_df.fillna(-100)

train_df = wifi_df.head(int(len(wifi_df)*0.8))
test_df = wifi_df.tail(int(len(wifi_df)*0.2))

test_index = list(test_df.index)


testing_data_path = os.path.join(derived_data_path, 'testing_data')
if not os.path.exists(testing_data_path):
    os.mkdir(testing_data_path)
with open(os.path.join(testing_data_path, 'test_data_index_2021.json'), 'w+') as file:
    json.dump(test_index, file)

X_train = train_df.iloc[:,:FEATURE_LEN].astype(float).values
# X_train = train_df.iloc[:,:FEATURE_LEN].apply(lambda x:pow(10, x/10)).astype(float).values

print('X_train', X_train.shape)
Y_train = train_df.loc[:, ['x', 'y']].astype(float).values

X_test = test_df.iloc[:,:FEATURE_LEN].astype(float).values
# X_test = test_df.iloc[:,:FEATURE_LEN].apply(lambda x:pow(10, x/10)).astype(float).values

Y_test = test_df.loc[:, ['x', 'y']].astype(float).values

knn = KNeighborsRegressor(n_neighbors=3)

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

'''
[[139.63016202  87.06910134]
 [235.65254694 107.15941097]
 [ 82.29603225 124.03844663]]
[[145.12     80.68852]
 [235.11621 103.13599]
 [ 87.30561 120.60939]]
4.918635045529956

[[139.63016202  87.06910134]
 [235.65254694 107.15941097]
 [ 82.29603225 124.03844663]]
[[145.12     80.68852]
 [235.11621 103.13599]
 [ 87.30561 120.60939]]
4.918635045529956
'''