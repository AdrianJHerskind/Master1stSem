from os import listdir
from os.path import isfile, join
import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import stitching.stitching as st

plt.rcParams["figure.figsize"] = (20, 15)  # Plot it BIG

# spark (ours)
# intrinsic = np.mat([[2.87086548e+03, 0.00000000e+00, 2.01225464e+03],
#                            [0.00000000e+00, 2.86824438e+03, 1.44638072e+03],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# distCoeffs = np.array([ 2.48150268e-01, -1.10089398e+00,  5.27836952e-04,  3.99240690e-03, 1.60390228e+00])

# other team
intrinsic = np.mat([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
distCoeffs = np.array([0.117, -0.298, 0.001, 0, 0.1420])

camera = {"s_size": [6.3, 4.7], "calib":
    {"cameraMatrix": intrinsic,
     "distCoeffs": distCoeffs}}

e_time = time.time()
stitcher = st.Stitcher(camera, 5, plot=True)

mypath = '../fieldImages/dronePicturesField1'  # change this path as necessary
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles[:]:
    stitcher.add_img(file)

image, gps_centre, offset = stitcher.process_images()
e_time = time.time() - e_time

print("Picture stitched in {}s".format(e_time))

imageRGB = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)
cv.imwrite("result.png", image)
plt.imshow(imageRGB)
plt.title("Stitched Image :D")
plt.show()
