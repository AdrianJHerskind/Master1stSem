import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2 as cv
import random
from stitching.helpers import get_grouping_order

mypath = "35m/"
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

gps = []
n = -1

for path_and_file in onlyfiles[:n]:
    path, file = os.path.split(path_and_file)
    pre, ext = os.path.splitext(file)
    temp_path = os.path.join(path, "temp")
    pickle_path = os.path.join(temp_path, "%s.pickle" % pre)
    image = pickle.load(open(pickle_path, 'rb'))
    gps.append(image[3])

images = [(1, 1, 1, g, 1) for g in gps]

data = pickle.load(open('35m/temp/matches.pickle', 'rb'))
data = data[0:n, 0:n]
order = get_grouping_order(images, data)
for a in order:
    print(a)
