import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('calib/*.JPG')
gray = None
for fname in images:
    print("Reading image {}".format(fname))
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("Finding the chessboard corners")
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6))

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Found chessboard corners")
        objpoints.append(objp)

        print("Computing subpixel location of corners")
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        print("Drawing the corners")
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.imshow(img2)
        plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(ret)
print(mtx)
print(dist)
cv2.destroyAllWindows()
