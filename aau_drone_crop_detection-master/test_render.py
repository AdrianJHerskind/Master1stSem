import pickle

import numpy as np
from matplotlib import pyplot as plt
import stitching.stitching as st
import cv2 as cv
import matplotlib.axis

# plt.rcParams["figure.figsize"] = (20, 15)  # Plot it BIG

camera = {"s_size": (6.3, 4.7), "calib":
    {"cameraMatrix": np.mat([[2.87086548e+03, 0.00000000e+00, 2.01225464e+03],
                             [0.00000000e+00, 2.86824438e+03, 1.44638072e+03],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
     "distCoeffs": np.array([2.48150268e-01, -1.10089398e+00, 5.27836952e-04, 3.99240690e-03, 1.60390228e+00])}}

res = 5
stitcher = st.Stitcher(camera, res)

images = pickle.load(open("stitched_images.pickle", "rb"))

image, centre, offset = stitcher.render(images)

imageRGB = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)

cv.imwrite("result.png", image)

fig, ax = plt.subplots()
ax.set_title("Stitched Image :D")
shape = np.array(imageRGB.shape[:2]) * (res / 100)
step = 20
shape = step + shape - np.mod(shape, step)
x_ticks = np.linspace(0, shape[1], 6)
y_ticks = np.linspace(0, shape[0], 6)
ax.imshow(imageRGB)
ax.set_xticks(np.round(x_ticks / (res / 100)))
ax.set_yticks(np.round(y_ticks / (res / 100)))
ax.set_xticklabels(x_ticks.astype("int"))
ax.set_yticklabels(y_ticks.astype("int"))

plt.show()
