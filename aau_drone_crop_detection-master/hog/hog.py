import cv2
import numpy as np

vKernel = np.mat([[-1], [0], [1]])
hKernel = np.mat([-1, 0, 1])


def get_hog_descr(img, binSize=20, blockSize=3):
    # Compute gradients
    hGradient = cv2.filter2D(img, -1, hKernel)
    vGradient = cv2.filter2D(img, -1, vKernel)

    # Get the angles and magnitude of the pixel gradients
    mags, angles = combine_gradients(hGradient, vGradient)

    # Helper variables
    numOfCells = blockSize * blockSize
    numOfBins = int(360 / binSize)
    cellHist = np.zeros((numOfCells, numOfBins))
    cellSize = int(mags.shape[0] / blockSize)

    # Fill in the weighted votes
    h = mags.shape[0]
    w = mags.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            cell = int(x / cellSize) + int(y / cellSize) * blockSize
            angle = angles[x, y]
            lowbin = int(angles[x, y] / binSize)

            highratio = angles[x, y] % binSize / binSize
            cellHist[cell, (lowbin if lowbin < numOfBins else 0)] += (1 - highratio) * mags[x, y]
            cellHist[cell, (lowbin + 1 if lowbin + 1 < numOfBins else 0)] += highratio * mags[x, y]

    cellHist = cellHist / np.linalg.norm(cellHist.ravel())

    # Return average?
    # Block size bigger than feature?
    # ??¿??¿
    average = np.zeros((numOfBins))
    for i in range(0, numOfBins):
        for j in range(0, numOfCells):
            average[i] += cellHist[j, i]
        average[i] /= numOfCells

    # Return Value
    ret = dict()
    ret["hist"] = average
    ret["binSize"] = binSize
    ret["numOfBins"] = numOfBins
    ret["numOfCells"] = numOfCells
    ret["blockDimension"] = blockSize
    ret["cellDimension"] = cellSize
    # return average
    return cellHist[4, :]


def combine_gradients(hGradient, vGradient):
    h = hGradient.shape[0]
    w = hGradient.shape[1]

    arrayHGradient = hGradient.ravel()
    arrayVGradient = vGradient.ravel()

    mag, angle = cv2.cartToPolar(arrayHGradient.astype(float), arrayVGradient.astype(float), angleInDegrees=True)

    return mag.reshape(h, w), angle.reshape(h, w)


if __name__ == "__main__":
    # sample execution
    print("Running sample")
    img = cv2.imread("../hahog/aau1.jpg", cv2.IMREAD_GRAYSCALE)
    result = np.zeros(img.shape + (2,))
    get_hog_descr(img, (500, 500), 30)
