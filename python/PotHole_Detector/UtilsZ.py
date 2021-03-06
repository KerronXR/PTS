import os
import time
import numpy as np
import cv2.cv2 as cv
import pickle
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_path = 'C:/Users/Roman/Documents/Python/Project/Data/VideoCrop/'
new_dir = image_path + str(time.time())
os.mkdir(new_dir)

path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
modelName = 'ResNet50V2_128x128_1780_Detection_Chance=[0.8901]_2020-06-03 12_55_18.130516.tflite'
IMAGE_SIZE = (int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0]))

interpreter = tf.lite.Interpreter(model_path=path + modelName)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

thresh_val_list = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV]
adaptive_thresh_val_list = [cv.ADAPTIVE_THRESH_MEAN_C, cv.ADAPTIVE_THRESH_GAUSSIAN_C]


def do_nothing(x):
    pass


def detect_holes(grey_frame, original_frame, trackbars, count):
    thresh = cv.adaptiveThreshold(grey_frame, *trackbars)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        # approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        # area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if len(contour) > 100 and 100 < perimeter < 300:
            # if len(contour) > 5 and 200 < area < 1000:
            x, y, w, h = cv.boundingRect(contour)
            img_c = original_frame[y:y + h, x:x + w]
            # if count % 30 == 0:
            #     cv.imwrite(new_dir + '/' + str(time.time()) + str(random()) + '.jpg', img_c)
            img = cv.resize(img_c, IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Converting into 4 dimension array
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            if predictions[0][0] > 0.9:
                # print(predictions[0][1])
                # if count % 30 == 0:
                cv.imwrite(new_dir + '/' + str(time.time()) + str(random()) + '.jpg', img_c)
                cv.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv.drawContours(original_frame, [contour], 0, (0, 255, 0), 2)
    return original_frame


def undistort(img, cal_dir='cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv.undistort(img, mtx, dist, None, mtx)
    return dst


def color_filter(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    result = cv.meanStdDev(hsv)
    print(result[1][2])
    lower_white = np.array([0, 0, int(result[1][2]) + 100])
    upper_white = np.array([255, 255, 255])
    masked_white = cv.inRange(hsv, lower_white, upper_white)
    return masked_white


def thresholding(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5))
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv.Canny(img_blur, 50, 100)
    img_close = cv.morphologyEx(img_canny, cv.MORPH_GRADIENT, np.ones((10, 10)))
    img_dial = cv.dilate(img_canny, kernel, iterations=1)
    img_erode = cv.erode(img_dial, kernel, iterations=1)
    img_color = color_filter(img)
    combined_image = cv.bitwise_or(img_color, img_erode, img_close)

    return combined_image, img_canny, img_color, img_close


def initialize_trackbars(initial_trackbar_values):
    cv.namedWindow("Adaptive Threshold")
    cv.resizeWindow("Adaptive Threshold", 360, 240)
    cv.createTrackbar('thresh_val', 'Adaptive Threshold', initial_trackbar_values[0], 1, do_nothing)
    cv.createTrackbar('adapt_val', 'Adaptive Threshold', initial_trackbar_values[1], 1, do_nothing)
    cv.createTrackbar("high_threshold", 'Adaptive Threshold', initial_trackbar_values[2], 10, do_nothing)
    cv.createTrackbar("block_size", 'Adaptive Threshold', initial_trackbar_values[3], 101, do_nothing)
    cv.createTrackbar("sub_const", 'Adaptive Threshold', initial_trackbar_values[4], 100, do_nothing)


def val_trackbars():
    thresh_val = cv.getTrackbarPos('thresh_val', "Adaptive Threshold")
    adapt_val = cv.getTrackbarPos('adapt_val', "Adaptive Threshold")
    high_threshold = cv.getTrackbarPos("high_threshold", "Adaptive Threshold")
    block_size = cv.getTrackbarPos("block_size", "Adaptive Threshold")
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    sub_const = cv.getTrackbarPos("sub_const", "Adaptive Threshold")
    src = [high_threshold, thresh_val_list[thresh_val], adaptive_thresh_val_list[adapt_val],
           block_size, sub_const]
    return src


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def find_road_area(grey_frame, original_frame, grey_copy):
    lines = cv.HoughLinesP(grey_frame, 1, np.pi / 180, 35, minLineLength=80, maxLineGap=200)
    # img = np.copy(original_frame)
    # blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_left = [0, 0, 0, 0, 0, 0]
    line_right = [0, 0, 0, 0, 0, 0]
    roi_vertices = [(0, 0), (0, 0), (0, 0)]
    found_left = False
    found_right = False
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                m = ((y2 - y1) / (x2 - x1))
                if 0.7 > m > 0.3:
                    found_right = True
                    # cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
                    b = y1 - (m * x1)
                    # line_right = [x1, y1, ((original_frame.shape[0] * 0.94 - b) / m) - 20,
                    #               original_frame.shape[0] * 0.94, m, b]
                    line_right = [x1, y1, x2 + 10, y2, m, b]
                    if found_right and found_left:
                        break
                if -0.7 < m < -0.3:
                    found_left = True
                    # cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
                    b = y1 - (m * x1)
                    # line_left = [((original_frame.shape[0] * 0.92 - b) / m) + 20,
                    #              original_frame.shape[0] * 0.92, x2, y2, m, b]
                    line_left = [x1 + 20, y1, x2, y2, m, b]
                    if found_right and found_left:
                        break
    # original_frame = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    if found_right and found_left:
        if line_left != line_right:  # if both the same line - pass, else do:
            mx = line_right[4] - line_left[4]
            if mx != 0:  # if no intersections - pass, else do:
                b = line_left[5] - line_right[5]
                if b == 0:
                    cross_x = 0
                else:
                    cross_x = int(b / mx)
                cross_y = int(line_right[4] * cross_x + line_right[5])
                roi_vertices = [(line_left[0], line_left[1]), (cross_x, cross_y), (line_right[2], line_right[3])]
        # roi_vertices = [(line_left[0], line_left[1]), (line_left[2], line_left[3]),
        #                 (line_right[0], line_right[1]), (line_right[2], line_right[3])]
    else:
        print('Lane Not Found')
        cv.putText(original_frame, 'Lane Not Found',
                   (int(original_frame.shape[1] / 2 - 100), int(original_frame.shape[0] / 2)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    roi_frame = region_of_interest(grey_copy, np.array([roi_vertices], np.int32))
    return roi_frame, original_frame
