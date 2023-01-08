
import copy
from shapely.geometry import shape, GeometryCollection, Polygon, MultiPolygon
from shapely.affinity import affine_transform
from PIL import Image, ImageOps
import nudged
import numpy as np
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import json
import itertools
import os

site = 'KDM'
data_root_path = '../imputation/data/'
data_path = os.path.join(data_root_path, site)

geojson_file = '../imputation/data/%s/topo/geojson_map.json' % (site)
infos_file = '../imputation/data/%s/topo/floor_info.json' % (site)
image_file = '../imputation/data/%s/topo/floor_image.png' % (site)

image = Image.open(image_file)
with open(infos_file, 'rb') as f:
    info = json.load(f)
with open(geojson_file, 'rb') as f:
    geojson = json.load(f)

def extract_coords_from_polygon(polygon):
    coords = []
    if type(polygon) == MultiPolygon:
        polygons = polygon.geoms
    else:
        polygons = [polygon]

    for polygon in polygons:
        x, y = polygon.exterior.xy
        coords.append((np.array(x), np.array(y)))
        for interior in polygon.interiors:
            x, y = interior.xy
            coords.append((np.array(x), np.array(y)))
    return coords

def get_bounding_box(x, y):
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    return np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max]
    ])

def plot_shape(shapes):
    if type(shapes) == Polygon:
        shapes = [shapes]
    for shape in shapes:
        for interior in shape.interiors:
            plt.plot(*interior.xy)
        plt.plot(*shape.exterior.xy)

def plot_shape(shapes, color):
    if type(shapes) == Polygon:
        shapes = [shapes]
    for shape in shapes:
        plt.fill(*shape.exterior.xy, color='white')
        for interior in shape.interiors:
            # plt.plot(*interior.xy, c=color)
            plt.fill(*interior.xy, color='whitesmoke')
            plt.plot(*interior.xy, c='black', linewidth=0.5)
        plt.plot(*shape.exterior.xy, c='black', linewidth=0.5)


def plot_points(points,color,size, zorder):
    markers = []
    # marker = itertools.cycle((',', '+', '.', 'o', '*'))
    for point in points:
        plt.scatter(*point, c=color, s=size, zorder=zorder)#edgecolors='black', linewidths=0.1

def extract_geometries(geojson):
    # Extract floor plan geometry (First geometry)
    floor = copy.deepcopy(geojson)
    floor['features'] = [floor['features'][0]]
    floor_layout = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in floor['features']])[0]

    # Extract shops geometry (remaining ones)
    shops = copy.deepcopy(geojson)
    shops['features'] = shops['features'][1:]
    shops_geometry = GeometryCollection([shape(feature["geometry"]).buffer(0.1) for feature in shops['features']])

    # Geometry differences to get corridor (floor layout - shops)
    corridor = copy.deepcopy(floor_layout)
    for shop in shops_geometry:
        corridor = corridor.difference(shop)
    return floor_layout, corridor


def extract_image_bounding_box(image):
    # Flip and convert to black and white
    gray_image = ImageOps.flip(image).convert('LA')
    bw_image = np.array(gray_image.point(lambda p: p > 251 and 255)) > 0
    bw_image = Image.fromarray(bw_image.any(axis=2) == True)

    # Get convex hull
    ch_image = convex_hull_image(np.array(bw_image))

    # Transform to coordinates
    image_y, image_x = np.where(ch_image == True)

    bounding_box = get_bounding_box(image_x, image_y)
    return bounding_box


def extract_geojson_bounding_box(floor_layout):
    # Get convex hull
    ch_geojson = floor_layout.convex_hull

    coords = [coord for coord in ch_geojson.exterior.coords]
    geojson_x = [coord[0] for coord in coords]
    geojson_y = [coord[1] for coord in coords]

    bounding_box = get_bounding_box(geojson_x, geojson_y)
    return bounding_box


def find_translation(points_a, points_b):
    trans = nudged.estimate(points_a, points_b)
    matrix_cooefs = np.ravel(trans.get_matrix())

    trans_coeffs = [
        matrix_cooefs[0],
        matrix_cooefs[1],
        matrix_cooefs[3],
        matrix_cooefs[4],
        matrix_cooefs[2],
        matrix_cooefs[5],
    ]

    return trans_coeffs


def geo_referencing():
    # Extract floor layout and corridor geometries from geojson (shapely Polygon/MultiPolygon)
    floor_layout, corridor = extract_geometries(geojson)
    # Extract bounding boxes both from image and geojson (Using convexhull)
    image_bounding_box = extract_image_bounding_box(image)
    geojson_bounding_box = extract_geojson_bounding_box(floor_layout)

    # Find best translation from geojson to image referential
    translation_coeffs = find_translation(geojson_bounding_box, image_bounding_box)

    # Convert to image size scale
    translated_corridor = affine_transform(corridor, translation_coeffs)

    # Convert to waypoints scale (using ratio between waypoint scale and image scale)
    x_ratio = info["map_info"]["width"] / image.size[0]
    y_ratio = info["map_info"]["height"] / image.size[1]
    waypoint_translation_coeffs = [
        x_ratio, 0, 0,
        y_ratio, 0, 0
    ]
    translated_corridor = affine_transform(translated_corridor, waypoint_translation_coeffs)
    print('translated_corridor', translated_corridor)
    return translated_corridor


