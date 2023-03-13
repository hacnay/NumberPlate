import os
import shutil

import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import regionprops


def capture_frames(video_filename, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_filename)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('window-name', frame)
            cv2.imwrite(f"{output_dir}/frame{count:d}.jpg", frame)
            count += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def load_image(filename):
    car_image = imread(filename, as_gray=True)
    car_image = imutils.rotate(car_image, 270)
    gray_car_image = np.uint8(car_image * 255)
    return car_image, gray_car_image


def binarize_image(image):
    threshold_value = threshold_otsu(image)
    binary_car_image = image > threshold_value
    return binary_car_image


def extract_candidate_regions(binary_image):
    label_image = measure.label(binary_image)

    plate_dimensions = (
        0.03 * label_image.shape[0],
        0.08 * label_image.shape[0],
        0.15 * label_image.shape[1],
        0.3 * label_image.shape[1]
    )

    plate_dimensions2 = (
        0.08 * label_image.shape[0],
        0.2 * label_image.shape[0],
        0.15 * label_image.shape[1],
        0.4 * label_image.shape[1]
    )

    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []

    for region in regionprops(label_image):
        if region.area < 50:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_image[min_row:max_row, min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

    if not plate_like_objects:
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []

        for region in regionprops(label_image):
            if region.area < 50:
                continue

            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col

            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                plate_like_objects.append(binary_image[min_row:max_row, min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

    return plate_like_objects, plate_objects_cordinates


def plot_candidate_regions(image, regions, cordinates):
    fig, ax = plt.subplots(1)
    ax
