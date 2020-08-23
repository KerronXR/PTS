import os
import time
import cv2.cv2 as cv
from random import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

image_path = 'C:/Users/Roman/Documents/Python/Project/Data/VideoCrop/'
new_dir = image_path + str(time.time())
os.mkdir(new_dir)

path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
modelName = 'PotHoleDetector_128x128_.tflite'
IMAGE_SIZE = (int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0]))

interpreter = tf.lite.Interpreter(model_path=path + modelName)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


def detect_holes(grey_frame, original_frame, frame_copy):
    thresh = cv.adaptiveThreshold(grey_frame, 1, 0, 0, 61, 10)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        # area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if len(contour) > 100 and 100 < perimeter < 300:
            # if len(contour) > 5 and 200 < area < 1000:
            x, y, w, h = cv.boundingRect(contour)
            img_c = frame_copy[y:y + h, x:x + w]
            img = cv.resize(img_c, IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Converting into 4 dimension array
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            if predictions[0][0] > 0.9:
                cv.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.imwrite(new_dir + '/' + str(time.time()) + str(random()) + '.jpg', img_c)
    return original_frame


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def find_road_area(grey_frame, original_frame, grey_copy):
    lines = cv.HoughLinesP(grey_frame, 1, 3.14159265359 / 180, 35, minLineLength=80, maxLineGap=200)
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
        cv.putText(original_frame, 'Lane Not Found',
                   (int(original_frame.shape[1] / 2 - 100), int(original_frame.shape[0] / 2)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    roi_frame = region_of_interest(grey_copy, np.array([roi_vertices], np.int32))
    return roi_frame, original_frame
