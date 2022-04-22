# coding: utf-8
import math
import numpy as np
from geo_analyze import geo_referencing,plot_points, plot_shape
from shapely.geometry import shape, GeometryCollection, Polygon, MultiPolygon, MultiPoint
import pandas as pd


def euler_distance(point1: np.ndarray, point2: list) -> float:
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)
class ClusterNode(object):
    def __init__(self, vec, pts, left=None, right=None, distance=-1, id=None, count=1):
        self.vec = vec
        self.pts = pts
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count
class Hierarchical(object):
    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None
        self.geometry = geo_referencing()

    def fit(self, x):
        nodes = [ClusterNode(vec=v[:-2], id=i, pts=tuple(v[-2:])) for i,v in enumerate(x)]
        distances = {}
        point_num, future_num = np.shape(x)  # 特征的维度
        future_num = future_num - 2
        self.labels = [ -1 ] * point_num
        currentclustid = -1
        min_dist = -1
        while min_dist < math.inf:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None  # 表示最相似的两个聚类
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances:
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if (d < min_dist) and self.is_able_to_be_clustered(nodes[i], nodes[j]):
                        min_dist = d
                        closest_part = (i, j)
            if min_dist < math.inf:
                part1, part2 = closest_part
                node1, node2 = nodes[part1], nodes[part2]
                new_vec = [(node1.vec[i] * node1.count + node2.vec[i] * node2.count) / (node1.count + node2.count)
                            for i in range(future_num)]
                new_node = ClusterNode(vec=new_vec,
                                       pts=None,
                                       left=node1,
                                       right=node2,
                                       distance=min_dist,
                                       id=currentclustid,
                                       count=node1.count + node2.count)
                currentclustid -= 1
                del nodes[part2], nodes[part1]   # 一定要先del索引较大的
                nodes.append(new_node)
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        for i, node in enumerate(self.nodes):
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        if node.left == None and node.right == None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

    def is_able_to_be_clustered(self, node_i, node_j):
        collected_pts_i = []
        collected_pts_j = []

        collected_pts_i = self.node_traversal(node_i, collected_pts_i)
        collected_pts_j = self.node_traversal(node_j, collected_pts_j)

        collect_pts = collected_pts_i + collected_pts_j

        # see if it contains null location
        collect_pts = list(filter(lambda x: x[0]!=np.nan and x[1]!=np.nan, collect_pts))

        pts_polygon = MultiPoint(collect_pts).convex_hull
        intersected_polygon = self.geometry.intersection(pts_polygon)
        if intersected_polygon.area == pts_polygon.area:
            return True
        else:
            return False

    def node_traversal(self, node, collected_points):

        if node.left == None and node.right == None:
            collected_points.append(node.pts)
            # print('len of collected_points', len(collected_points))
        if node.left:
            collected_points = self.node_traversal(node.left, collected_points)
        if node.right:
            collected_points = self.node_traversal(node.right, collected_points)
        return collected_points

def find_waypoint(t, wp_df):
    wp_df = wp_df.sort_values(by=['wp_ts'], ascending=True)
    wp_df_a = wp_df.shift(-1).rename(columns={'wp_ts':'a_ts', 'x':'a_x', 'y':'a_y'})
    wp_df_c = pd.concat([wp_df, wp_df_a], axis=1)
    target_row = wp_df_c.loc[(wp_df_c['wp_ts']<=t)&(wp_df_c['a_ts']>t),['wp_ts','x','y','a_ts', 'a_x', 'a_y']].reset_index(drop=True)
    if not target_row.empty:
        x = target_row.loc[0,'x'] + ((target_row.loc[0,'a_x']-target_row.loc[0,'x'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        y = target_row.loc[0,'y'] + ((target_row.loc[0,'a_y']-target_row.loc[0,'y'])/(target_row.loc[0,'a_ts']-target_row.loc[0,'wp_ts']))*(t-target_row.loc[0,'wp_ts'])
        # print('target_row', x, y)
        return x, y
    else:
        wp_df = wp_df.drop_duplicates(subset=['wp_ts', 'x', 'y'])
        wp_df = wp_df.reset_index(drop=True)
        if len(wp_df)>=2:
            if t <= wp_df.loc[0,'wp_ts']:
                # print('t',t)
                # print('wp_df', wp_df)
                x = wp_df.loc[0,'x'] - ((wp_df.loc[1, 'x'] - wp_df.loc[0, 'x']) * (wp_df.loc[0,'wp_ts'] - t))/(wp_df.loc[1, 'wp_ts'] - wp_df.loc[0, 'wp_ts'])
                y = wp_df.loc[0,'y'] - ((wp_df.loc[1, 'y'] - wp_df.loc[0, 'y']) * (wp_df.loc[0,'wp_ts'] - t))/(wp_df.loc[1, 'wp_ts'] - wp_df.loc[0, 'wp_ts'])
                # print('approach 111', x,y)
            else:
                x = ((wp_df.loc[len(wp_df)-1,'x'] - wp_df.loc[len(wp_df)-2,'x'])*(t - wp_df.loc[len(wp_df)-1,'wp_ts']))/(wp_df.loc[len(wp_df)-1,'wp_ts'] - wp_df.loc[len(wp_df)-2,'wp_ts']) +  wp_df.loc[len(wp_df)-1,'x']
                y = ((wp_df.loc[len(wp_df)-1,'y'] - wp_df.loc[len(wp_df)-2,'y'])*(t - wp_df.loc[len(wp_df)-1,'wp_ts']))/(wp_df.loc[len(wp_df)-1,'wp_ts'] - wp_df.loc[len(wp_df)-2,'wp_ts']) +  wp_df.loc[len(wp_df)-1,'y']
                # print('approach 222', x, y)
        else:
            # print('wp_df', wp_df)
            if t <= wp_df.loc[0,'wp_ts']:
                x = wp_df.loc[0,'x'] - (wp_df.loc[0,'wp_ts']-t)*1
                y = wp_df.loc[0,'y'] - (wp_df.loc[0,'wp_ts']-t)*1
                # print('approach 333', x, y)
            else:
                x = (t - wp_df.loc[0,'wp_ts'])*1 + wp_df.loc[0,'x']
                y = (t - wp_df.loc[0,'wp_ts'])*1 + wp_df.loc[0,'y']
                # print('approach 444', x, y)
        return x, y


def interpolate_rp(group_df):
    ori_shape = group_df.shape
    ori_index = group_df.index
    have_wp_df = group_df.loc[~group_df['wp_ts'].isnull(), :]
    # print('have_wp_df shape', have_wp_df.shape)
    wp_df = have_wp_df.loc[:, ['wp_ts', 'x', 'y']]
    have_null_wp_df = group_df.loc[group_df['wp_ts'].isnull(), :]
    # print('have_null_wp_df shape', have_null_wp_df.shape)
    if not have_null_wp_df.empty:
        for index, row in have_null_wp_df.iterrows():
            ts = int(row['ts'])
            x, y = find_waypoint(ts, wp_df)
            if x and y:
                have_null_wp_df.loc[index, 'x'] = x
                have_null_wp_df.loc[index, 'y'] = y
                have_null_wp_df.loc[index, 'wp_ts'] = ts
            else:
                print('Hahahahah')
    # have_null_wp_df_after_inter = have_null_wp_df.loc[have_null_wp_df['wp_ts'].isnull(),:]
    # print('have_null_wp_df_after_inter shape', have_null_wp_df_after_inter.shape)
    group_df = pd.concat([have_wp_df, have_null_wp_df], axis=0)
    group_df = group_df.loc[ori_index,:]
    post_shape = group_df.shape
    # print(ori_shape, post_shape)
    assert (ori_shape[0]==post_shape[0]) & (ori_shape[1]==post_shape[1]), 'error'
    ip_x_y = group_df.loc[:, :]
    return ip_x_y





