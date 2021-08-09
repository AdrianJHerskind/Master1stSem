import cv2 as cv
import numpy as np
from PIL import Image

gaussianMask = np.ones((1, 1))


def getGaussianMask(size):
    global gaussianMask
    a, b = gaussianMask.shape
    if a == size == b:
        return gaussianMask

    sigma = (size >> 1) / 3
    kernelY = cv.getGaussianKernel(size, sigma)
    kernelY = kernelY / max(kernelY)
    gaussianMask = kernelY * cv.transpose(kernelY)
    return gaussianMask


def warpAffine(im, r, c, U, size):
    M = np.zeros((2, 3))
    a11 = U[1, 1]
    a12 = U[0, 1]
    a21 = U[1, 0]
    a22 = U[0, 0]
    halfSize = size >> 1
    M[0, 0] = a11
    M[1, 0] = a21
    M[0, 1] = a12
    M[1, 1] = a22
    M[0, 2] = c - a12 * halfSize - a11 * halfSize
    M[1, 2] = r - a22 * halfSize - a21 * halfSize
    return cv.warpAffine(im, M, (size, size), flags=cv.WARP_INVERSE_MAP + cv.INTER_LINEAR)


def img_blend(src1, src2):
    # compute the size of the panorama
    im2 = Image.fromarray(src2)
    im1 = Image.fromarray(src1)
    nw, nh = map(max, im2.size, im1.size)

    # paste img1 on top of img2
    newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg1.paste(im2, (0, 0))
    newimg1.paste(im1, (0, 0))

    # paste img2 on top of img1
    newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg2.paste(im1, (0, 0))
    newimg2.paste(im2, (0, 0))

    # blend with alpha=0.5
    result = Image.blend(newimg1, newimg2, alpha=0.5)
    open_cv_image = np.array(result)
    return open_cv_image


def gaussianBlur(img, sigma):
    size = int(2.0 * 3.0 * sigma + 1.0)
    if size % 2 == 0:
        size = size + 1
    return cv.GaussianBlur(img, (size, size), sigma, None, sigma, cv.BORDER_REPLICATE)


def localize(r, c, cur, high, low):
    drr = cur[r - 1, c] - 2.0 * cur[r, c] + cur[r + 1, c]
    dcc = cur[r, c - 1] - 2.0 * cur[r, c] + cur[r, c + 1]
    dll = low[r, c] - 2.0 * cur[r, c] + high[r, c]
    drc = 0.25 * (cur[r + 1, c + 1] - cur[r + 1, c - 1] - cur[r - 1, c + 1] + cur[r - 1, c - 1])
    drl = 0.25 * (high[r + 1, c] - high[r - 1, c] - low[r + 1, c] + low[r - 1, c])
    dcl = 0.25 * (high[r, c + 1] - high[r, c - 1] - low[r, c + 1] + low[r, c - 1])

    edgeScore = (drr + dcc) * (drr + dcc) / (drr * dcc - drc * drc)

    # Hessian matrix: H(x)
    H = np.mat([[drr, drc, drl],
                [drc, dcc, dcl],
                [drl, dcl, dll]])
    dr = .5 * (cur[r + 1, c] - cur[r - 1, c])
    dc = .5 * (cur[r, c + 1] - cur[r, c - 1])
    dl = .5 * (high[r, c] - low[r, c])
    # Gradient: -∇f(x)
    g = np.mat([[dr], [dc], [dl]])
    try:
        delta_x = np.linalg.solve(H, -g)
        val = cur[r, c] + 0.5 * (dr * delta_x[0, 0] + dc * delta_x[1, 0] + dl * delta_x[2, 0])
        return delta_x, val, edgeScore, True
    except np.linalg.LinAlgError:  # If solution is not valid solve throws this error
        return [], 0, edgeScore, False


def hessianResponse(img, norm):
    Lxx = cv.filter2D(img, -1, np.mat([1, -2, 1]))
    Lyy = cv.filter2D(img, -1, np.mat([[1], [-2], [1]]))
    Lxy = cv.filter2D(img, -1, np.mat([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / 4)
    response = (Lxx * Lyy - Lxy * Lxy) * pow(norm, 2)

    return response


def isMax(val, img, row, col):
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if img[r, c] > val:
                return False
    return True


def isMin(val, img, row, col):
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if img[r, c] < val:
                return False
    return True


def halfImage(img):
    out = img[1:2:]
