import PIL.ExifTags as Exif
import cv2 as cv
from PIL import Image
from geopy import distance as geodist
import matplotlib.pyplot as plt
import numpy as np
import heapq
import networkx as nx
import math
import random


def read_image(file):
    img = Image.open(file)
    exif = img._getexif()

    exif_data = {}
    for tag, value in exif.items():
        decoded = Exif.TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = Exif.GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]

            exif_data[decoded] = gps_data
        else:
            exif_data[decoded] = value

    cvimg = np.array(img)
    cvimg = np.float32(cvimg)
    cvimg *= (1 / 255.0)
    return cvimg, exif_data


def get_data_from_exif(exif):
    fLength = exif["FocalLength"]
    fLength = fLength[0] / fLength[1]
    gps = (exif["GPSInfo"]["GPSLatitude"], exif["GPSInfo"]["GPSLongitude"])
    gps = (gps[0][0][0] / gps[0][0][1] + gps[0][1][0] / gps[0][1][1] / 60 + gps[0][2][0] / gps[0][2][1] / 3600,
           gps[1][0][0] / gps[1][0][1] + gps[1][1][0] / gps[1][1][1] / 60 + gps[1][2][0] / gps[1][2][1] / 3600)
    height = exif["GPSInfo"]["GPSAltitude"]
    height = height[0] / height[1]

    return fLength, height, gps


def adjust_spatial_res(image, target_res, height, s_size, f_length):
    current_sr = s_size[0] * height / (image.shape[0] * f_length) * 100  # Compensates the units
    down_ratio = current_sr / target_res
    small_img = cv.resize(image,
                          (0, 0),  # set fx and fy, not the final size
                          fx=down_ratio,
                          fy=down_ratio,
                          interpolation=cv.INTER_AREA)
    return small_img


def get_neighbour_features(gps, radius, images):
    """
Get features of images that are close to a gps location
    :param gps: gps location (a tuple with (lat, long))
    :param radius: radius where we want the features (in meters)
    :param images: vector of tuples (image, position, features, gps)
    :return: features of the images with gps within the radius, (key points, descriptors)
    """
    kps = []
    dscrs = []
    for image in images:
        if geodist.great_circle(image[4], gps).meters < radius:
            kps.extend(image[3][0])
            dscrs.extend(image[3][1])

    return np.array(kps), np.array(dscrs)


def compute_features(img):
    """
    Compute the AKAZE features of the given image
    :param img: image
    :return: The keypoints and descriptors of the features
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 0, 1, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    mask = cv.erode(mask, kernel, iterations=2)
    mask = np.array(mask, np.uint8)

    feature_extractor = cv.AKAZE.create()
    kps, dscrs = feature_extractor.detectAndCompute(gray, mask)

    new_idx = np.argsort([kp.response for kp in kps])[::-1][:5000]
    kps = [kps[i] for i in new_idx]
    dscrs = [dscrs[i] for i in new_idx]

    print("Found %i features" % len(kps))
    return kps, np.float32(dscrs)


def get_radius(shape, res):
    gps_error = 20  # m // maybe this should be calculated but we lack the data to do so
    image_radius = math.sqrt(shape[0] ** 2 + shape[1] ** 2) / 2 * res / 100
    return gps_error + image_radius


def compute_match_matrix(images, res):
    n = len(images)
    matches = np.zeros((n, n))
    for i in range(n):
        small_path, shape, (_, new_descrs), gps, _ = images[i]
        radius = get_radius(shape, res)
        print("Matching %s" % small_path)
        for j in range(n):
            _, _, (_, old_descrs), gps_j, _ = images[j]
            if i < j and geodist.great_circle(gps_j, gps).meters < radius:
                mat = match_features(new_descrs[:750], old_descrs[:750])
                matches[i][j] = len(mat)
                matches[j][i] = len(mat)
    return matches


def get_graph(matches):
    match_threshold = 50
    G, weight = get_graph_threshold(matches, match_threshold)
    while not nx.algorithms.is_connected(G) and match_threshold > 0:
        match_threshold = match_threshold - 5
        G, weight = get_graph_threshold(matches, match_threshold)
    print("Generated graph with match threshold: %i" % match_threshold)
    return G, weight


def get_graph_threshold(matches, match_threshold):
    rows, cols = np.where(matches > match_threshold)
    weight = (match_threshold / matches[rows, cols])
    edges = zip(rows.tolist(), cols.tolist(), weight.tolist())
    G = nx.Graph()
    G.add_nodes_from(range(matches.shape[0]))
    G.add_weighted_edges_from(edges)
    return G, weight


def plot_graph(G, weight, images):
    n = len(images)
    gps = [g for (_, _, _, g, _) in images]
    lat_mean = np.mean([g[0] for g in gps])
    long_mean = np.mean([g[1] for g in gps])
    lat_len, long_len = get_lat_long_lens(lat_mean)
    pos = {i: [(gps[i][0] - lat_mean) * lat_len,
               (gps[i][1] - long_mean) * long_len] for i in range(n)}
    nx.draw(G, pos,
            with_labels=True,
            width=weight * 20,
            node_size=1200)
    plt.show()


def plot_grouping_order(order):
    n = len(order) + 1
    images = {(tuple([a]), 0) for a in range(n)}
    nodes = []
    for (a, b) in order:
        for x in list(images):
            if all(i in a for i in x[0]):
                xa = x
                images.remove(x)
            if all(i in b for i in x[0]):
                xb = x
                images.remove(x)

        step = 1
        if len(xa[0]) == 1 and len(xb[0]) != 1:
            step = 0

        xc = (xa[0] + xb[0], max(xa[1], xb[1]) + step)
        nodes.append((xa, xb, xc))
        # drawing = cv.line(drawing, (xa[1], xa[2]), (xa[1], xc[2]), 0)
        # drawing = cv.line(drawing, (xb[1], xb[2]), (xb[1], xc[2]), 0)
        # drawing = cv.line(drawing, (xa[1], xc[2]), (xb[1], xc[2]), 0)

        images.add(xc)

    order, highest_level = xc
    rever = np.argsort(order)

    mult_x = 16
    mult_y = 8
    drawing = np.ones((mult_y * n, mult_x * (highest_level + 1))) * 255
    for (na, xa), (nb, xb), (nc, xc) in nodes:
        ya = np.average([rever[i] for i in na])
        yb = np.average([rever[i] for i in nb])
        xa = int((xa + .5) * mult_x)
        xb = int((xb + .5) * mult_x)
        xc = int((xc + .5) * mult_x)
        ya = int((ya + .5) * mult_y)
        yb = int((yb + .5) * mult_y)
        drawing = cv.line(drawing, (xa, ya), (xc, ya), 0)
        drawing = cv.line(drawing, (xb, yb), (xc, yb), 0)
        drawing = cv.line(drawing, (xc, ya), (xc, yb), 0)
    # cv.imshow("win", drawing)

    # imageRGB = cv.cvtColor(drawing, cv.COLOR_RGBA2BGRA)
    drawing = cv.merge((drawing, drawing, drawing))
    plt.imshow(drawing)
    plt.show()
    # cv.waitKey()

    return order


def get_grouping_order(images, G):
    n = len(images)

    splits = nx.algorithms.community.girvan_newman(G)

    last_split = next(splits)
    order = [last_split]

    for new_split in splits:
        p = next((bi, ai.difference(bi)) for (ai, bi) in zip(last_split, new_split) if ai != bi)
        if len(p[0]) > len(p[1]):
            p = (p[1], p[0])
        order.append(p)
        last_split = new_split

    order.reverse()

    # uncomment to check validity of order
    images = {frozenset([a]) for a in range(n)}
    for (a, b) in order:
        images.remove(frozenset(a))
        images.remove(frozenset(b))
        images.add(frozenset(a.union(b)))
    assert (images == {frozenset(range(n))})
    return order


def match_features(query_descrs, base_descrs):
    """
    Gives the set of good matches of the query features into the given feature set
    :param query_descrs: The features to be matched
    :param base_descrs: The feature set to match the query features into
    :return: The good matches (distance ratio 0.75)
    """
    # Brute force match (more efficient with low amount of features
    # Probably worth to change to a FLANN matcher (review)
    # matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    if len(base_descrs) == 0 or len(query_descrs) == 0:
        return []

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(query_descrs, base_descrs, 2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches


def compute_homography(query_kps, base_kps, matches):
    query_pts = np.float32([query_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    base_pts = np.float32([base_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv.findHomography(query_pts, base_pts, cv.RANSAC)
    return M


def get_image_size(images):
    maxD = [0., 0.]
    minD = [1E10, 1E10]
    for im in range(len(images)):
        if images[im] is not None:
            (_, image_shape, _, _, M) = images[im]
            corners = np.zeros((4, 2))
            corners[0][0] = 0
            corners[0][1] = 0

            corners[1][0] = 0
            corners[1][1] = image_shape[0]

            corners[2][0] = image_shape[1]
            corners[2][1] = 0

            corners[3][0] = image_shape[1]
            corners[3][1] = image_shape[0]

            transformed_corners = cv.perspectiveTransform(np.array([corners]), M)
            transformed_corners = transformed_corners[0]
            for transformed_corner in transformed_corners:
                maxD[0] = np.max([transformed_corner[0], maxD[0]])
                maxD[1] = np.max([transformed_corner[1], maxD[1]])
                minD[0] = np.min([transformed_corner[0], minD[0]])
                minD[1] = np.min([transformed_corner[1], minD[1]])

    maxD = np.ceil(maxD)
    minD = np.floor(minD)
    return (int(maxD[0] - minD[0]), int(maxD[1] - minD[1])), \
           (int(-minD[0]), int(-minD[1]))


def img_blend(l_img, s_img, t):
    [h, w, _] = np.array(list(s_img.shape))
    [x, y] = np.round(np.array(list(t)))

    tmp = cv.cvtColor(s_img, cv.COLOR_BGR2GRAY)
    _, alpha_s = cv.threshold(tmp, 0, 1, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    alpha_s = np.float32(alpha_s)
    alpha_s = cv.erode(alpha_s, kernel, iterations=1)
    alpha_s = cv.GaussianBlur(alpha_s, (0, 0), 20)
    alpha_s = cv.normalize(alpha_s, alpha_s, 0., 1., cv.NORM_MINMAX)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    alpha_s = cv.erode(alpha_s, kernel, iterations=4)
    f_black = np.ceil(s_img[:, :, 1] / 255.)
    f_black = cv.erode(f_black, kernel, iterations=2)
    alpha_s *= f_black

    black = 1 - cv.cvtColor(l_img[y:y + h, x:x + w, :3], cv.COLOR_BGR2GRAY) / 255.
    mask = np.array(black - alpha_s, np.int32)
    mask[mask < 1] = 0

    alpha_l = 1 - alpha_s

    for c in range(0, 3):
        s_img_comp = (1 - mask) * (alpha_s * s_img[:, :, c] + alpha_l * l_img[y:y + h, x:x + w, c])
        l_img_comp = mask * s_img[:, :, c]
        l_img[y:y + h, x:x + w, c] = s_img_comp + l_img_comp

    l_img[y:y + h, x:x + w, 3] = (1 - mask) * 255 + (mask * l_img[y:y + h, x:x + w, 3])

    return l_img


def get_lat_long_lens(lat):
    # compute length in meters of a degree of latitude and longitude
    # from https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula/75535
    lat = math.radians(lat)

    m1 = 111132.92  # latitude calculation term 1
    m2 = -559.82  # latitude calculation term 2
    m3 = 1.175  # latitude calculation term 3
    m4 = -0.0023  # latitude calculation term 4
    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3

    lat_len = m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) + (m4 * math.cos(6 * lat))
    long_len = (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) + (p3 * math.cos(5 * lat))

    return lat_len, long_len


def showImage(windowname, oriimg):
    cv.destroyWindow(windowname)
    H, W = 600, 600
    height, width = oriimg.shape[:2]

    scaleWidth = float(W) / float(width)
    scaleHeight = float(H) / float(height)
    if scaleHeight > scaleWidth:
        imgScale = scaleWidth
    else:
        imgScale = scaleHeight

    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv.resize(oriimg, (int(newX), int(newY)))
    cv.imshow(windowname, newimg)


def orient_north(images, res, plot=False):
    gps_points = []
    image_points = []
    for i in range(len(images)):
        if images[i] is not None:
            (_, shape, _, gps, M) = images[i]
            center = np.array([[list(shape[:2])]]) / 2.0
            [[t_center]] = cv.perspectiveTransform(center, M)
            gps_points.append(gps)
            image_points.append(t_center)

    lat_mean = np.mean([g[0] for g in gps_points])
    long_mean = np.mean([g[1] for g in gps_points])
    lat_len, long_len = get_lat_long_lens(lat_mean)
    desired_points = [[(gps[1] - long_mean) * long_len * 100 / res,
                       -(gps[0] - lat_mean) * lat_len * 100 / res] for gps in gps_points]

    image_points = np.float32(image_points).reshape(-1, 1, 2)
    desired_points = np.float32(desired_points).reshape(-1, 1, 2)
    M1 = np.eye(3)
    M1[:2, :], _ = cv.estimateAffinePartial2D(image_points, desired_points, method=cv.LMEDS)
    M2 = np.eye(3)
    M2[:2, :], _ = cv.estimateAffine2D(image_points, desired_points, method=cv.LMEDS)
    M3, _ = cv.findHomography(image_points, desired_points, method=cv.LMEDS)

    if M3 is not None and np.linalg.eigvals(M3[:2, :2])[0].real > 0:
        M = M3
        print("Orienting with findHomography")
    elif not np.any(np.isnan(M2)) and np.linalg.eigvals(M2[:2, :2])[0].real > 0:
        M = M2
        print("Orienting with estimateAffine2D")
    elif not np.any(np.isnan(M1)) and np.linalg.eigvals(M1[:2, :2])[0].real > 0:
        M = M1
        print("Orienting with estimateAffinePartial2D")
    else:
        print("Failed to North-orient the image")
        M = np.eye(3)
    if plot:
        transformed_image_points = cv.perspectiveTransform(image_points, M)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(desired_points[:, 0, 0], desired_points[:, 0, 1], c='r')
        ax1.scatter(image_points[:, 0, 0], image_points[:, 0, 1], c='b')
        ax1.scatter(transformed_image_points[:, 0, 0], transformed_image_points[:, 0, 1], c='g')
        ax1.legend(('desired_points', 'image_points', 'transformed_image_points'))
        plt.show()

    return M, (lat_mean, long_mean)


def compensate_offset(M, offset):
    M1 = M.copy()
    xoffset = M1[0][2] + offset[0]
    yoffset = M1[1][2] + offset[1]
    M1[0][2] = 0
    M1[1][2] = 0
    return M1, (xoffset, yoffset)


def rotateImage(image, angle):
    row, col, _ = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv.warpAffine(image, rot_mat, (col, row))
    return new_image


def get_sharp_score(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray_norm = cv.equalizeHist(gray)
    score = cv.Laplacian(gray, cv.CV_32F).var()
    score = -10 * math.log10(score)
    return score


def get_get_pixel_gps(pixel, centre, offset, resolution):
    lat_len, lon_len = get_lat_long_lens(centre[0])
    lat_pixel = centre[0] - ((pixel[1] - offset[1]) * resolution / 100) / lat_len
    lon_pixel = centre[1] + ((pixel[0] - offset[0]) * resolution / 100) / lon_len
    return lat_pixel, lon_pixel
