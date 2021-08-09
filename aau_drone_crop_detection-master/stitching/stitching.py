import copyreg
import json
import os
import pickle
import sys
from os import listdir
from os.path import join, isfile

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import cv2 as cv
import numpy as np
from geopy import distance as geodist
import argparse
import ctypes

from stitching import helpers

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                         point.response, point.octave, point.class_id)


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


def _pickle_match(match):
    return cv.KeyPoint, (match.queryIdx, match.trainIdx, match.distance)


copyreg.pickle(cv.DMatch().__class__, _pickle_match)


class Stitcher:

    # Initializer / Instance Attributes
    def __init__(self, camera, resolution, plot=False):
        self.default_camera = camera
        self.images = []  # (small_path, size, (keypoints, descriptors), gps)
        self.stitched_images = []  # (i, small_path, size, (keypoints, descriptors), gps, H)
        self.target_res = resolution  # cm per pixel
        self.timing = {}  # Timing statistics
        self.plots = plot

    def add_img(self, path_and_file, camera=None, min_shapness=0):
        # Process paths
        path, file = os.path.split(path_and_file)
        pre, ext = os.path.splitext(file)

        self.temp_path = os.path.join(path, "temp")
        small_path = os.path.join(self.temp_path, "small_%s.jpg" % pre)
        pickle_path = os.path.join(self.temp_path, "%s.pickle" % pre)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        if os.path.exists(pickle_path):
            print("loading %s" % pickle_path)
            res = pickle.load(open(pickle_path, 'rb'))
        else:
            print("no pickle found, processing %s" % path_and_file)
            if camera is None:
                camera = self.default_camera
            # - Read the image and EXIF
            (image, exif) = helpers.read_image(path_and_file)
            print("Image {} read".format(file))
            # - Discard the image if its too blurry
            score = helpers.get_sharp_score(image)
            if score < min_shapness:
                print("Image {} discarded. Too blurry, sharpness: {}, minimum: {}".format(file, score, min_shapness))
                return
            # - Get GPS and focal length data from EXIF data from image
            (f_length, height, gps) = helpers.get_data_from_exif(exif)
            og = image.copy()
            image = np.array(image * 255, dtype=np.uint8)
            img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
            # equalize the histogram of the Y channel
            img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
            # # convert the YUV image back to RGB format
            image = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
            image = np.array(image / 255., dtype=np.float32)
            # - Rectify image using camera calibration parameters
            undistorted = cv.undistort(image, camera["calib"]["cameraMatrix"], camera["calib"]["distCoeffs"])
            og_undistorted = cv.undistort(og, camera["calib"]["cameraMatrix"], camera["calib"]["distCoeffs"])
            # - Downsample the image to the required spatial resolution
            small_img = helpers.adjust_spatial_res(undistorted, self.target_res, height, camera["s_size"], f_length)
            og_small_img = helpers.adjust_spatial_res(og_undistorted, self.target_res, height, camera["s_size"],
                                                      f_length)
            cv.imwrite(small_path, cv.cvtColor(np.uint8(og_small_img * 255), cv.COLOR_RGB2BGR))

            # - Compute the features of the new image
            new_kps, new_descrs = helpers.compute_features(small_img)

            res = (small_path, og_small_img.shape, (new_kps, new_descrs), gps, np.eye(3))

            pickle.dump(res, open(pickle_path, 'wb'))
        self.images.append(res)

    def get_match_matrix(self):
        pickle_path = os.path.join(self.temp_path, "matches.pickle")
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        n = len(self.images)
        if os.path.exists(pickle_path):
            print("loading %s" % pickle_path)
            matches = pickle.load(open(pickle_path, 'rb'))
            if len(matches) >= n:
                return matches[0:n, 0:n]

        print("Loading unsuccessful, recalculating")
        matches = helpers.compute_match_matrix(self.images, self.target_res)
        pickle.dump(matches, open(pickle_path, 'wb'))
        return matches

    def process_images(self):
        matches_matrix = self.get_match_matrix()
        G, weight = helpers.get_graph(matches_matrix)
        if self.plots:
            helpers.plot_graph(G, weight, self.images)
        order = helpers.get_grouping_order(self.images, G)
        if self.plots:
            helpers.plot_grouping_order(order)
        for im_ord, (set_query, set_base) in enumerate(order):
            print("Computing homography %i of images %s to %s" % (im_ord, set_query, set_base))
            reduced_set_query, reduced_set_base = self.reduce_sets(set_query, set_base, matches_matrix)
            (query_descrs, query_kps) = self.get_features_from_set(reduced_set_query)
            (base_descrs, base_kps) = self.get_features_from_set(reduced_set_base)
            matches = helpers.match_features(query_descrs, base_descrs)
            if len(matches) <= 10:
                print("Sets failed to match features")
                for i in set_query:
                    self.images[i] = None
                continue
            M = helpers.compute_homography(query_kps, base_kps, matches)
            for i in set_query:
                self.remap_features(i, M)
        return self.render(self.images)

    def reduce_sets(self, set1, set2, matches):
        new_set1 = set()
        new_set2 = set()
        for i in set1:
            for j in set2:
                if matches[i, j] > 10:
                    new_set1.add(i)
                    new_set2.add(j)

        if len(new_set1) == 0 or len(new_set2) == 0:
            new_set1 = set1
            new_set2 = set2

        return new_set1, new_set2

    def get_features_from_set(self, set):
        kps = []
        descrs = []
        for i in set:
            if self.images[i] is not None:
                (_, _, (kps_i, descrs_i), _, _) = self.images[i]
                kps.extend(kps_i[::int(1 + (len(set) / 4))])
                descrs.extend(descrs_i[::int(1 + (len(set) / 4))])
        return np.array(descrs), np.array(kps)

    def render(self, images):
        pickle.dump(images, open("stitched_images.pickle", 'wb'))
        print("Rendering the map with {} images...".format(len(images)))
        # Render stitched image from image stack, return image and metadata
        # TODO: NEED ORIENTATION OF THE IMAGE
        Rot, centre = helpers.orient_north(images, self.target_res, plot=self.plots)
        for i in range(len(images)):
            if images[i] is not None:
                (xa, xb, xc, xd, M) = images[i]
                M = Rot.dot(M)
                images[i] = (xa, xb, xc, xd, M)
        shape, offset = helpers.get_image_size(images)
        print("Total size of the stitched picure: {}x{}px".format(shape[0], shape[1]))
        cummulative_img = np.full((shape[1], shape[0], 4), 0, dtype=np.uint8)
        i = 0
        for im in reversed(range(len(images))):
            if images[im] is not None:
                (img, _, _, _, M) = images[im]
                i += 1
                print("Stitching {}/{} ...".format(i, len(images)), end="\r")
                img = cv.imread(img)
                sm_shape, sm_offset = helpers.get_image_size([(_, img.shape, _, _, M)])
                M1 = M.copy()
                M2 = np.eye(3)
                M2[0][2] += sm_offset[0]
                M2[1][2] += sm_offset[1]

                transformed = cv.warpPerspective(img, M2.dot(M1), sm_shape)

                cummulative_img = helpers.img_blend(cummulative_img, transformed,
                                                    np.array(list(offset))
                                                    - np.array(list(sm_offset)))

        cummulative_img[:, :, 3] = cv.GaussianBlur(cummulative_img[:, :, 3], (0, 0), 20)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        cummulative_img[:, :, 3] = cv.erode(cummulative_img[:, :, 3], kernel, iterations=4)
        print("Rendering finished")

        return cummulative_img, centre, offset

    def remap_features(self, img_idx, M):
        if self.images[img_idx] is not None:
            (img, shape, (kps, dscrs), gps, H) = self.images[img_idx]
            # - Transform the features of the new picture and add them to the cumulative feature set
            new_kps = kps
            for i in range(len(kps)):
                kp = kps[i]
                p = np.array([[[kp.pt[0], kp.pt[1]]]])
                pp = cv.perspectiveTransform(p, M)
                new_kps[i].pt = (pp[0][0][0], pp[0][0][1])
            self.images[img_idx] = (img, shape, (new_kps, dscrs), gps, M.dot(H))


    @property
    def __img_no__(self):
        return len(self.images)


def stitch_folder(path, camera, spatial_res):
    stitcher = Stitcher(camera, spatial_res)
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for file in onlyfiles:
        stitcher.add_img(file)
    image, gps = stitcher.process_images()
    return image, gps


def stitch_folderJSON(path, paramsJson):
    params = json.loads(paramsJson)
    return stitch_folder(path, params["Camera"], params["SpatialRes"])


def stitch_and_save_folderJSON(path, paramsJson, dest):
    params = json.loads(paramsJson)
    params["Camera"]["calib"]["cameraMatrix"] = np.array(params["Camera"]["calib"]["cameraMatrix"]).reshape((3, 3))
    params["Camera"]["calib"]["distCoeffs"] = np.array(params["Camera"]["calib"]["distCoeffs"])
    image, gps = stitch_folder(path, params["Camera"], params["SpatialRes"])
    if not os.path.exists(dest):
        os.makedirs(dest)
    cv.imwrite(os.path.join(dest, "orthomap.png"), image)
    json.dump({"gps": list(gps)}, open(os.path.join(dest, "gps_data.json"), "w"))
    return image


def run():
    intrinsic = [2951, 0, 1976, 0, 2951, 1474, 0, 0, 1]
    distCoeffs = [0.117, -0.298, 0.001, 0, 0.1420]
    camera = {"s_size": [6.3, 4.7], "calib": {"cameraMatrix": intrinsic, "distCoeffs": distCoeffs}}
    config = {"SpatialRes": 2.5, "Camera": camera}

    parser = argparse.ArgumentParser(description="Stitches images from a folder into an orthomap")
    parser.add_argument("--inputDir", "-i", help="Folder containing the images to be stitched", required=True)
    parser.add_argument("--output", "-o", help="Directory of the result image and JSON", required=False,
                        default="Result/")
    parser.add_argument("--config", "-c", help="Json file or string containing the configuration", required=False,
                        default=json.dumps(config))
    args = parser.parse_args()
    if isfile(args.config):
        with open(args.config, 'r') as configFile:
            args.config = configFile.read()
    stitch_and_save_folderJSON(args.inputDir, args.config, args.output)


if __name__ == "__main__":
    run()
