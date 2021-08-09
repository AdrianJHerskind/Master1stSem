import pickle

import numpy as np
from matplotlib import pyplot as plt
import stitching.helpers as helpers
import cv2 as cv
import matplotlib.axis

images = pickle.load(open("stitched_images.pickle", "rb"))
img, center, offset = pickle.load(open("output.pickle", "rb"))
lat_mean, lon_mean = center
lat_len, lon_len = helpers.get_lat_long_lens(lat_mean)

print(center)
map_img = cv.imread("map.png")
lat_min = 57.156
lat_max = 57.164
lon_min = 9.822
lon_max = 9.840
# map_img = cv.imread("map3.png")
# lat_min = 57.06
# lat_max = 57.063
# lon_min = 10.03
# lon_max = 10.037

H = np.eye(3)

# remove offset
H1 = np.eye(3)
H1[:2, 2] = (-offset[0], -offset[1])
H = H1.dot(H)

# scale
H1 = np.eye(3)
H1[0, 0] = 0.05 / ((lon_max - lon_min) * lon_len / map_img.shape[1])
H1[1, 1] = 0.05 / ((lat_max - lat_min) * lat_len / map_img.shape[0])
H = H1.dot(H)

# offset to gps
H1 = np.eye(3)
H1[0, 2] = (lon_mean - lon_min) / (lon_max - lon_min) * map_img.shape[1]
H1[1, 2] = (lat_max - lat_mean) / (lat_max - lat_min) * map_img.shape[0]
H = H1.dot(H)

w, h = map_img.shape[:2]
warp_img = cv.warpPerspective(img, H, (h, w))

H2 = np.eye(3)

map_img_add = map_img.copy()
alpha = warp_img[:, :, 3] / 255.0
for c in range(3):
    map_img_add[:, :, c] = map_img[:, :, c] * (1 - alpha) + warp_img[:, :, c] * alpha

cv.imwrite("map_result.png", map_img_add)
