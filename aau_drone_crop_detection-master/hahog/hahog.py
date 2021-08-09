from __future__ import print_function
import sys

sys.path.insert(0, '../hog')

import cv2 as cv
import numpy as np
import argparse
from PIL import Image
import operator
import math
import helpers
import time
import scipy.signal
import skimage.feature
import scipy.linalg
import pickle
import hog

akaze = cv.AKAZE_create()

# GLOBALS
firstBlur = []
prevBlur = []
blur = []
nextBlur = []
low = []
cur = []
high = []
pixelDistance = 1
octaveMap = []
gaussianMask = np.ones((1, 1))

params = {
    "upscaleInputImage": 0,
    "numberOfScales": 10,
    "initialSigma": 1.6,
    "threshold": 16 / 3,
    "edgeEigenValueRatio": 10,
    "border": 5,
    "maxSubpixelShift": .6,
    "maxIterations": 16,
    "desc_factor": 3 * math.sqrt(3),
    "convergenceThreshold": 0.05,
    "patchSize": 41,
    "verbose": False,
    "smmWindowSize": 19,
    "mrSize": 3.0 * math.sqrt(3.0),
    "descriptorWindowSize": 19 * 3
}

eevr = params["edgeEigenValueRatio"]
params["edgeScoreThreshold"] = (pow(eevr + 1, 2) / eevr)
params["finalThreshold"] = params["threshold"] * params["threshold"]
params["positiveThreshold"] = .8 * params["finalThreshold"]
params["negativeThreshold"] = -params["positiveThreshold"]


def findAffineShape(r, c, s, v):
    """
    Find the affine shape corresponding to the given point (r,c,s,v). The affine shape is given by the U matrix
    :return: r,c,s,U,v
    """
    global firstBlur, pixelDistance, params
    U = np.mat([[1, 0], [0, 1]])
    eigen_ratio_act = 0
    eigen_ratio_bef = 0
    lr = r / pixelDistance  # level r
    lc = c / pixelDistance  # level c
    ls = s / pixelDistance  # level s
    ratio = ls / params["initialSigma"]

    maskPixels = params["smmWindowSize"] * params["smmWindowSize"]
    halfSize = params["smmWindowSize"] >> 1

    for l in range(params["maxIterations"]):
        img = helpers.warpAffine(firstBlur, lr, lc, U * ratio, params["smmWindowSize"])

        Lx = cv.filter2D(img, -1, np.mat([-1, 0, 1]), borderType=cv.BORDER_REPLICATE)
        Ly = cv.filter2D(img, -1, np.mat([[-1], [0], [1]]), borderType=cv.BORDER_REPLICATE)
        mask = helpers.getGaussianMask(params["smmWindowSize"])

        A = Ly * Ly * mask
        B = Lx * Ly * mask
        C = Lx * Lx * mask

        smm = np.mat([[sum(sum(A)), sum(sum(B))], [sum(sum(B)), sum(sum(C))]])
        smm = smm / math.sqrt(np.linalg.det(smm))  # normalize

        mu = np.linalg.inv(scipy.linalg.sqrtm(smm))

        U = mu * U

        eigs_mu = np.linalg.eigvals(mu)
        eigs_U = np.linalg.eigvals(U)

        eigs_mu.sort()
        eigs_U.sort()

        eigen_ratio_bef = eigen_ratio_act
        eigen_ratio_act = 1 - eigs_mu[0] / eigs_mu[1]

        # new localization
        # levelBlur = helpers.gaussianBlur(firstBlur,
        #                                  math.sqrt(ls * ls - params["initialSigma"] * params["initialSigma"]))
        # deformedBlur = helpers.warpAffine(levelBlur, lr, lc, U, params["smmWindowSize"])
        # deformedResponse = helpers.hessianResponse(deformedBlur, ls)
        #
        # ir = halfSize
        # ic = halfSize
        # drr = deformedResponse[ir - 1, ic] - 2.0 * deformedResponse[ir, ic] + deformedResponse[ir + 1, ic]
        # dcc = deformedResponse[ir, ic - 1] - 2.0 * deformedResponse[ir, ic] + deformedResponse[ir, ic + 1]
        # drc = 0.25 * (deformedResponse[ir + 1, ic + 1] - deformedResponse[ir + 1, ic - 1] - deformedResponse[
        #     ir - 1, ic + 1] + deformedResponse[ir - 1, ic - 1])
        #
        # # Hessian matrix: H(x)
        # H = np.mat([[drr, drc],
        #             [drc, dcc]])
        #
        # dr = .5 * (deformedResponse[ir + 1, ic] - deformedResponse[ir - 1, ic])
        # dc = .5 * (deformedResponse[ir, ic + 1] - deformedResponse[ir, ic - 1])
        #
        # # Gradient: -∇f(x)
        # g = np.mat([[dr], [dc]])
        # try:
        #     #  recall that we were looking for: H(f(x))*Δx = -∇f(x)
        #     delta_ix = np.linalg.solve(H, -g)
        #     delta_x = U * delta_ix
        #     delta_r = delta_x[0, 0]
        #     delta_c = delta_x[1, 0]
        #     if abs(delta_r) > 1 or abs(delta_c) > 1:
        #         return None
        #     lr = lr + delta_r
        #     lc = lc + delta_c
        # except np.linalg.LinAlgError:  # If solution is not valid solve throws this error
        #     return None
        #
        # if np.any(np.iscomplex(eigs_U)):  # Should not happen
        #     return None

        if eigs_U[1] / eigs_U[0] > 6 * 6:  # Ellipse is too excentric
            return None

        if eigen_ratio_act < params["convergenceThreshold"] and eigen_ratio_bef < params[
            "convergenceThreshold"]:
            return lr * pixelDistance, lc * pixelDistance, s, U, v
    return None
    # exit()
    # image0 = img.copy() / 255
    # image0 = np.array([[[s, s, s] for s in r] for r in image0])
    # image0 = cv.resize(image0, (480, 480))
    # cv.imshow("Image", image0)
    # cv.waitKey()


def localizePoint(r, c, s):
    """
    Once we have found the a peak in (r,c,s), but those are integers. We now want to approximate the decimal value.
    To do so we do a taylor expansion of the function we are maximizing (the hessian response)
        f(x+Δx)  ~= f(x)+∇f(x)*Δx
    taking the derivative of this we get (see that we are looking for zero derivative)
        f'(x+Δx) ~= 0 = ∇f(x)+H(f(x))*Δx
                    H(f(x))*Δx = -∇f(x)
    :param r: row of the detected point in the level image
    :param c: col of the detected point in the level image
    :param s: scale of the detected point in the level image
    :return: final shape
    """
    global prevBlur, blur, nextBlur, low, cur, high, octaveMap, pixelDistance, params
    rows, cols = cur.shape[:2]

    converged = False
    val = cur[r, c]
    new_r = r
    new_c = c
    # In this loop what we will do is find the localization around the peak pixel and, if the shift is too big, 
    # change to a neighbor pixel and repeat
    for it in range(0, 5):
        r = new_r
        c = new_c

        # localization of the point
        delta_x, val, edge_score, status = helpers.localize(r, c, cur, high, low)

        if not status:
            # there has been an error in the localization
            return None

        if it == 0 and edge_score > params["edgeScoreThreshold"]:
            # the point looks too much like an edge, which is bad for hessian features, so we discard it
            return None

        delta_r = delta_x[0, 0]
        delta_c = delta_x[1, 0]
        delta_l = delta_x[2, 0]

        # if any of the delta_c is bigger than a threshold (0.6 pixels) we shift to the next pixel
        if delta_c > params["maxSubpixelShift"]:
            new_c = c + 1
        if delta_r > params["maxSubpixelShift"]:
            new_r = r + 1
        if delta_c < -params["maxSubpixelShift"]:
            new_c = c - 1
        if delta_r < -params["maxSubpixelShift"]:
            new_r = r - 1

        if not (params["border"] < c < cols - params["border"] and params["border"] < r < rows - params["border"]):
            # we have stepped to the border, discard point
            return None

        if new_r == r and new_c == c:
            converged = True
            break

    if abs(val) < params["finalThreshold"]:
        # the value is not above a threshold, discard the point
        return None

    if abs(delta_r) > 1.5 or abs(delta_c) > 1.5 or abs(delta_l) > .6:
        # the shift is still high (maybe it has not converged yet), we discard it
        return None

    if octaveMap[r, c] > 0:
        # we have already found this point on the octave, we discard it (this does not happen often)
        return None

    # Compute the row, column and scale in the original image
    final_r = (r + delta_r) * pixelDistance
    final_c = (c + delta_c) * pixelDistance
    final_s = s * pow(2, delta_l / params["numberOfScales"]) * pixelDistance

    affineShape = findAffineShape(final_r, final_c, final_s, val)

    if affineShape is None:
        # Now we mark it as found on the octave
        octaveMap[r, c] = 1

    return affineShape


def detectLevelPoints(levelSigma):
    """
    Finds points in current level, basically finding local maximum and minimum in the scale space
    :param levelSigma: the sigma of the level
    :return: points: the points of the level
    """
    print("Starting level %f" % (levelSigma * pixelDistance))
    global params, cur, low, high

    # Here we will store the points of the level
    points = []
    nPoints = 0

    # Find the points that are local maximum of the level, above a threshold and excluding the border
    for r, c in skimage.feature.peak_local_max(cur, threshold_abs=params["positiveThreshold"],
                                               exclude_border=params["border"]):
        val = cur[r, c]

        # check that it is also a local maximum in the scale space and not on the octave
        if helpers.isMax(val, low, r, c) and helpers.isMax(val, high, r, c):
            nPoints = nPoints + 1
            kp = localizePoint(r, c, levelSigma)
            if kp is not None:
                points.append(kp)

    # Find the points that are local minimum of the level, above a threshold and excluding the border
    for r, c in skimage.feature.peak_local_max(-cur, threshold_abs=-params["negativeThreshold"],
                                               exclude_border=params["border"]):
        val = cur[r, c]

        # check that it is also a local minimum in the scale space
        if helpers.isMin(val, low, r, c) and helpers.isMin(val, high, r, c):
            nPoints = nPoints + 1
            kp = localizePoint(r, c, levelSigma)
            if kp is not None:
                points.append(kp)
    print("Found %i points in the level, %i successfully localized" % (nPoints, len(points)))
    return points


def detectOctavePoints(firstLevel):
    """
    Detects the points on a single octave
    :param firstLevel: the first image of the octave
    :return:    points: the points of the octave
                lastLevel: the last Level of the octave
    """
    global pixelDistance, params, octaveMap, prevBlur, blur, nextBlur, low, cur, high, firstBlur

    # Here we store all points on the octave
    points = []

    # octaveMap[r,c] stores whether or not we have found a point in (r,c) in this octave, we only keep one per octave
    octaveMap = np.zeros_like(firstLevel)

    # At level we do sigma=sigma*sigmaStep. In order to get to the next octave we need sigmaStep^numberOfScales=2
    sigmaStep = math.pow(2, 1 / params["numberOfScales"])

    # For each level we want to compare it to the previous and next ones, so we need to store their info:
    # All of them are stored in global variables
    #        What      | previous  | current |   next
    # -----------------|-----------|---------|----------
    #   blurred image  | prevBlur  |  blur   | nextBlur
    # hessian response |    low    |   cur   |   high

    # Prepare first level
    levelSigma = params["initialSigma"]
    blur = firstLevel
    firstBlur = blur
    cur = helpers.hessianResponse(blur, pow(levelSigma, 2))

    # Start iterating
    # See that it goes up to i=numberOfScales+1, this is because the octave cannot detect points on its first level
    # and therefore the previous octave has to compute that level instead
    for i in range(1, params["numberOfScales"] + 2):

        # Compute next level
        needed_sigma = levelSigma * math.sqrt(sigmaStep * sigmaStep - 1.0)
        nextBlur = helpers.gaussianBlur(blur, needed_sigma)
        nextSigma = levelSigma * sigmaStep
        high = helpers.hessianResponse(nextBlur, nextSigma * nextSigma)

        # Detect points only if i>1 (if not we wouldn't have the previous level)
        if i > 1:
            points.extend(detectLevelPoints(levelSigma))

        # Save last level for next octave
        if i == params["numberOfScales"]:
            lastLevel = nextBlur.copy()

        # Shift variables
        prevBlur = blur
        blur = nextBlur
        low = cur
        cur = high
        levelSigma = nextSigma
    return points, lastLevel


def run_image(imgName):
    """
    Runs hahog detection for a single image
    :param imgName: name of the image to be detected
    """
    global pixelDistance

    # Process image
    imageColorOr = cv.imread(imgName)

    # imageColorOr = cv.resize(imageColorOr, (400, 320))
    imageColor = imageColorOr.astype(float)
    image = np.mean(imageColor, axis=2)  # convert to grayscale
    image0 = (image.copy() / 255)
    image0 = np.array([[[s, s, s] for s in r] for r in image0])

    # Starting pyramid (multi-scale feature detection)

    # prepare first level
    curSigma = .5  # we assume the sigma of the raw image is 0.5
    pixelDistance = 1

    initialSigma = params["initialSigma"]
    if initialSigma > curSigma:
        # blur image to get to desired initialSigma
        sigma = math.sqrt(initialSigma * initialSigma - curSigma * curSigma)
        octaveFirstLevel = helpers.gaussianBlur(image, sigma)
    else:
        octaveFirstLevel = image.copy()

    # set size where it will stop
    minSize = 2 * params["border"] + 2

    points = []  # here is where we will store all the points

    # iterate over all octaves
    while octaveFirstLevel.shape[0] > minSize and octaveFirstLevel.shape[1] > minSize:
        # detect points
        octaveKeypoints, octaveLastLevel = detectOctavePoints(octaveFirstLevel)
        points.extend(octaveKeypoints)

        # downsample
        octaveFirstLevel = octaveLastLevel[0::2, 0::2].copy()
        pixelDistance *= 2

    # Start visualization
    # select only top points (based on hessian response)
    top100 = sorted(points, key=lambda t: t[-1], reverse=True)[:1000]
    kp = []
    des = []
    for r, c, s, U, v in top100:
        # warp image for description
        dws = params["descriptorWindowSize"]
        img = helpers.warpAffine(image, r, c, U * s / (params["descriptorWindowSize"] / 3),
                                 params["descriptorWindowSize"])
        description = hog.get_hog_descr(img)
        # description = [0, 0, 0, 0]
        # Fold bins 0-180 with 180-360
        numBins = len(description)
        description = description[0:int(numBins / 2)] + description[int(numBins / 2):]

        # Get color from description
        if sum(description) == 0:
            hue = 0
        else:
            hue = sum(list(range(10, 180, 20)) * description) / sum(description)
        color = cv.cvtColor(np.uint8([[[hue * 255 / 180, 255, 255]]]), cv.COLOR_HSV2BGR)
        color = tuple(color[0, 0] / 255)

        # Draw ellipse from U matrix
        v, w = np.linalg.eig(U)
        ellipseAxis = (1 / np.sqrt(v)) * s
        # ellipseAxis = np.array([s,s])
        ellipseAngle = math.atan2(w[0, 0], w[1, 0]) * 180 / math.pi - 90
        image0 = cv.ellipse(image0, (int(c), int(r)), tuple(ellipseAxis.astype(int)), ellipseAngle, 0, 360, color, 2, cv.LINE_AA)
    # cv.imshow(imgName, image0)
    return (image0 * 255).astype("uint8")


def main():
    img = run_image('aau1.jpg')
    cv.imshow('aau1.jpg', img)
    cv.imwrite('aa1_hog.jpg', img)
    cv.waitKey()


def main_graf():
    for i in range(1, 7):
        imgName = 'graffiti/graf%i.png' % i
        print('Processing %s' % imgName)
        img = run_image(imgName)

        cv.imwrite('graffiti/hog%i.png' % i, img)

        if i != 1:
            H = np.loadtxt('graffiti/H1to%ip' % i)
            img = cv.warpPerspective(img, H, (int(img.shape[1]), int(img.shape[0])), flags=cv.WARP_INVERSE_MAP)

        # cv.imshow(imgName, img)
        cv.imwrite('graffiti/w_hog%i.png' % i, img)
        # cv.waitKey()


if __name__ == "__main__":
    start_time = time.time()
    main_graf()
    print("Runtime: %.2fs" % (time.time() - start_time))
