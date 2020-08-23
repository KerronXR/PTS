import os
import numpy as np
import cv2.cv2 as cv
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def find_intersections(frame, lines, height, width):
    # find the lines intersections to make a road triangle
    # find lines
    line_equations = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + (height + width) * (-b))
            y1 = int(y0 + (height + width) * a)
            x2 = int(x0 - (height + width) * (-b))
            y2 = int(y0 - (height + width) * a)
            # find the line equation
            m = ((y2 - y1) / (x2 - x1))
            b = y1 - (m * x1)
            line_equations.append([m, b])
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # find intersection points
    # mx1 - mx2 = b2 - b1
    intersection_points = []
    if line_equations is not None:
        for line1 in range(len(line_equations) - 1):
            for line2 in range(len(line_equations)):
                if line1 == line2:  # same line - pass
                    continue
                mx = line_equations[line1][0] - line_equations[line2][0]
                if mx == 0:  # no intersections - pass
                    continue
                b = line_equations[line2][1] - line_equations[line1][1]
                if b == 0:
                    cross_x = 0
                else:
                    cross_x = int(b / mx)
                cross_y = int(line_equations[line1][0] * cross_x + line_equations[line1][1])
                intersection_points.append((cross_x, cross_y))
                cv.circle(frame, (cross_x, cross_y), 10, [45, 215, 134], 3)

    # find triangle points
    triangle = [[0, 0], [0, 0], [0, 0]]
    average_left = [frame.shape[1] * 0.1, frame.shape[0] * 0.9]
    count_left = 1
    average_top = [frame.shape[1] * 0.5, frame.shape[0] * 0.1]
    count_top = 1
    average_right = [frame.shape[1] * 0.9, frame.shape[0] * 0.9]
    count_right = 1
    if intersection_points is not None:
        for intersection_point in intersection_points:
            # # only intersection points within the img range:
            if -1000 < intersection_point[0] < width * 2 and -1000 < intersection_point[1] < height:
                if intersection_point[1] > height * 0.25:
                    if intersection_point[0] > (width * 0.5):
                        average_right[0] += intersection_point[0]
                        average_right[1] += intersection_point[1]
                        count_right += 1
                    else:
                        average_left[0] += intersection_point[0]
                        average_left[1] += intersection_point[1]
                        count_left += 1
                else:
                    average_top[0] += intersection_point[0]
                    average_top[1] += intersection_point[1]
                    count_top += 1

    triangle[0][0] = int(average_left[0] / count_left)
    triangle[0][1] = int(average_left[1] / count_left)
    triangle[1][0] = int(average_top[0] / count_top)
    triangle[1][1] = int(average_top[1] / count_top)
    triangle[2][0] = int(average_right[0] / count_right)
    triangle[2][1] = int(average_right[1] / count_right)
    # triangle[0][0] = width * 0.1
    # triangle[0][1] = height * 0.45
    # triangle[1][0] = width * 0.5
    # triangle[1][1] = height * 0.1
    # triangle[2][0] = width * 0.9
    # triangle[2][1] = height * 0.45
    # cv.circle(img, (triangle[0][0], triangle[0][1]), 10, [45, 215, 134], 3)
    # cv.circle(img, (triangle[1][0], triangle[1][1]), 10, [45, 215, 134], 3)
    # cv.circle(img, (triangle[2][0], triangle[2][1]), 10, [45, 215, 134], 3)
    return triangle


def find_road_area(grey_frame, cut_frame, cut_frame_shape, cut_frame_ratio):
    edges = cv.Canny(grey_frame, 100, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=450)
    centered_line_left = [0, cut_frame_shape[1], int(cut_frame_shape[0] / 2), int(cut_frame_shape[1] / 3)]
    centered_line_right = [cut_frame_shape[0], cut_frame_shape[1], int(cut_frame_shape[0] / 2),
                           int(cut_frame_shape[1] / 3)]
    most_left_corner = (cut_frame_shape[0] + cut_frame_shape[1]) / (cut_frame_ratio * 2)
    most_right_corner = (cut_frame_shape[0] + cut_frame_shape[1]) - ((cut_frame_shape[0] + cut_frame_shape[1]) / (
            cut_frame_ratio * 2))
    # check if sums of axes is not more than
    # (cut_frame_shape[0] + cut_frame_shape[1]) / (cut_frame_ratio * 2) px far of the corners
    # choose the closest to corners line start points
    # if all lines are far away then use default centered_lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 + (cut_frame_shape[1] - y1) < most_left_corner:
                if x2 + y2 < cut_frame_shape[0]:
                    most_left_corner = x1 + (cut_frame_shape[1] - y1)
                    centered_line_left = line[0]
            if x2 + y2 > most_right_corner:
                if x1 > y1 * cut_frame_ratio:
                    most_right_corner = x2 + y2
                    centered_line_right = line[0]
            cv.line(cut_frame, (x1, y1), (x2, y2), (0, 175, 255), 2)
    roi_vertices = [(centered_line_left[0], centered_line_left[1]),
                    (centered_line_left[2], centered_line_left[3]),
                    (centered_line_right[2], centered_line_right[3]),
                    (centered_line_right[0], centered_line_right[1])]
    cv.line(cut_frame, (centered_line_left[0], centered_line_left[1]),
            (centered_line_left[2], centered_line_left[3]), (225, 227, 98), 2)
    cv.line(cut_frame, (centered_line_right[0], centered_line_right[1]),
            (centered_line_right[2], centered_line_right[3]), (227, 98, 102), 2)
    roi = region_of_interest(grey_frame, np.array([roi_vertices], np.int32))
    return roi
