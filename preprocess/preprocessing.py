import os
import time
import sys
sys.path.append('/Users/xiaol/PycharmProjects/WifiLocalization')
import pandas as pd
from io_f import read_data_file
from pathlib import Path
import numpy as np
from itertools import chain
import json

# floor df shape 1 (2342, 2502) site3F2
# floor df shape 1 (4104, 929) site5F4


data_root_path = '../data'
site = 'site5'
data_path = os.path.join(data_root_path, site)
delta_t = 10000

floor_dirs = Path(data_path).resolve().glob("./*/")
floor_bssids = {}
floor_dirs = list(filter(lambda x:str(x).split('/')[-1][0] != '.', floor_dirs))
derived_data_path = '../derived_data/{site}'.format(site=site)
with open(os.path.join(derived_data_path, 'wifi_bssids.json'), 'r+') as file:
    wifi_bssids = json.load(file)

print('wifi_bssids', len(wifi_bssids))

building_bssids = list(set(list(chain(*(wifi_bssids.values())))))
print(len(building_bssids))

st = time.time()

# def aggregate(wifi_blocks_dict):
#     '''
#     :param wifi_blocks_path: (timestamp, wifi_block_df)
#     :return:
#     '''
#     timestamps = list(wifi_blocks_dict.keys())
#     print(timestamps)
#     min_ts = min(timestamps) - 10000
#     max_ts = max(timestamps) + 10000
#     clusters = kde_cluster(timestamps, min_ts, max_ts)
#     return clusters

wifi_fp_building = []
for floor_dir in floor_dirs:
    path_files_total = Path(floor_dir).resolve().glob("./*/")
    path_files_total = list(path_files_total)
    path_files_str = list(filter(lambda x: str(x).split('/')[-1] == "path_data_files", list(path_files_total)))[0]
    path_files = Path(path_files_str).resolve().glob("./*/")
    path_files = list(path_files)
    print('len of path_files',len(path_files))
    bssid_floor = []
    floor_num = str(path_files_str).split('/')[-2]
    if not ((site=='site4' and floor_num=='F1') or (site=='site5' and floor_num=='F4') ):
        continue
    print('site, floor:', site, floor_num)
    wifi_fp_floor = []
    cnt = 0
    wp_total_list = []
    group_reuse_counter = 0
    for path_filename in path_files:
        path_data = read_data_file(path_filename)
        wifi_data = path_data.wifi
        waypoint = path_data.waypoint
        wifi_df = pd.DataFrame(wifi_data)
        if wifi_df.empty:
            continue
        wifi_df.columns = ['timestamp', 'ssid', 'bssid', 'rssi', 'last_ts']
        wifi_df = wifi_df.loc[abs(wifi_df['last_ts'].astype(int) - wifi_df['timestamp'].astype(int)) < delta_t,:]  # see if the threshold can be learned
        wifi_df['timestamp'] = wifi_df['timestamp'].astype(int)
        wifi_df['last_ts'] = wifi_df['last_ts'].astype(int)
        wifi_df['rssi'] = wifi_df['rssi'].astype('float')
        groups = wifi_df.groupby('timestamp')
        used_groups_index = []
        wp_list = []
        for wp in waypoint:
            td_list = []
            group_list = []
            name_list = []
            wp_total_list.append((wp[1], wp[2]))
            for name, group in groups:
                group_df = pd.DataFrame(group[['bssid', 'rssi']])
                group_df = group_df.drop_duplicates(subset=['bssid']).reset_index(drop=True)
                group_list.append(group_df)
                td = abs(wp[0] - int(name))
                td_list.append(td)
                name_list.append(name)
            matched_index = np.argmin(td_list)
            # if matched_index in used_groups_index:
            #     continue
            if td_list[matched_index] > 1000:
                continue
            wp_list.append(wp)
            group_df = group_list[matched_index]
            used_groups_index.append(matched_index)
            group_df = group_df.set_index(['bssid']).reindex(wifi_bssids[floor_num], fill_value=np.nan).T  # why -999 is set
            group_df['floor'] = floor_num
            group_df['x'] = wp[1]
            group_df['y'] = wp[2]
            group_df['wp_ts'] = wp[0]
            group_df['ts'] = name_list[matched_index]
            group_df['path'] = str(path_filename).split('/')[-1].split('.')[0]
            group_df = group_df.reset_index(drop=True)
            wifi_fp_floor.append(group_df)
        if len(used_groups_index)!=len(set(used_groups_index)):
            group_reuse_counter += 1
            print('used_groups_index', used_groups_index)
            print('wp_list', wp_list)
        # assert len(used_groups_index)==len(set(used_groups_index)), 'Repeated group index matched!'
        for index, (name, group) in enumerate(groups):
            group_df = pd.DataFrame(group[['bssid', 'rssi']])
            group_df = group_df.drop_duplicates(subset=['bssid']).reset_index(drop=True)
            if index in used_groups_index:
                continue
            else:
                wp_matched = [np.nan, np.nan, np.nan]# if no reference points can be matched, then just leave the reference point nan
            group_df = group_df.set_index(['bssid']).reindex(wifi_bssids[floor_num], fill_value=np.nan).T  # why -999 is set
            group_df['floor'] = floor_num
            group_df['x'] = wp_matched[1]
            group_df['y'] = wp_matched[2]
            group_df['wp_ts'] = wp_matched[0]
            group_df['ts'] = int(name)
            group_df['path'] = str(path_filename).split('/')[-1].split('.')[0]
            group_df = group_df.reset_index(drop=True)
            wifi_fp_floor.append(group_df)
        cnt += 1
        print('path cnt', cnt)
    # print('wp_total_list', len(wp_total_list))
    # wp_total_unique_list = list(set(wp_total_list))
    # print('wp_total_unique_list',len(wp_total_unique_list))

    print('{floor}:group_reuse_counter'.format(floor=floor_num), group_reuse_counter)
    wifi_floor_df = pd.concat(wifi_fp_floor, axis=0).reset_index(drop=True)

    # wifi_floor_df_null = wifi_floor_df.loc[wifi_floor_df['wp_ts'].isnull(),:]
    # wifi_floor_df_non_null = wifi_floor_df.loc[~wifi_floor_df['wp_ts'].isnull(),:]
    #
    # wifi_floor_df_non_null = wifi_floor_df_non_null.drop_duplicates(subset=['x', 'y']).reset_index(drop=True)
    # wifi_floor_df = pd.concat([wifi_floor_df_null, wifi_floor_df_non_null], axis=0).sort_values(by=['path', 'ts']).reset_index(drop=True)
    floor_sample_path = '../derived_data/{site}/{floor}'.format(site=site, floor=floor_num)
    if not os.path.exists(floor_sample_path):
        os.makedirs(floor_sample_path)
    print('wifi_floor_df before drop duplicates', wifi_floor_df.loc[~wifi_floor_df['wp_ts'].isnull(),:].shape)
    print('wifi_floor_df after drop duplicates', wifi_floor_df.loc[~wifi_floor_df['wp_ts'].isnull(),:].drop_duplicates(subset=['x','y']).shape)

    wifi_floor_df.to_csv(os.path.join(floor_sample_path, 'fp_sample_{floor}.csv'.format(floor=floor_num)), header=True, index=False)

    # total_len = len(wifi_floor_df)
    # null_wp_len = len(wifi_floor_df.loc[wifi_floor_df['wp_ts'].isnull(),:])
    # print(total_len, null_wp_len, round(null_wp_len/total_len, 4))
    # path_number = wifi_floor_df['path'].unique().tolist()
    # print('path number', len(path_number))
    # path_groups = wifi_floor_df.groupby(['path']).agg(lambda x:x.count())
    # print(path_groups.loc[:, ['x', 'y', 'wp_ts', 'ts']])
    # print(path_groups[['wp_ts','ts']].describe())

    print('floor df shape 1', wifi_floor_df.shape)
    side_info = wifi_floor_df.loc[:,['floor', 'x', 'y', 'wp_ts', 'ts', 'path']]
    wifi_floor_df = wifi_floor_df.drop(['floor', 'x', 'y', 'wp_ts', 'ts', 'path'], axis=1)

    wifi_floor_df = wifi_floor_df.T.reindex(building_bssids, fill_value=-100).T
    wifi_floor_df = pd.concat([wifi_floor_df, side_info], axis=1)
    print('floor df shape 2', wifi_floor_df.shape)
    wifi_fp_building.append(wifi_floor_df)

wifi_building_df = pd.concat(wifi_fp_building, axis=0).reset_index(drop=True)
building_sample_path = '../derived_data/{site}'.format(site=site)
if not os.path.exists(building_sample_path):
    os.makedirs(building_sample_path)
# wifi_building_df.to_csv(os.path.join(building_sample_path, 'fp_sample_building.csv'), header=True, index=False)

print('elapsed time(m)', (time.time()-st)/60)



