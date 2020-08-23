import time
import numpy as np
import cv2.cv2 as cv
from PotHole_Detector import Utils

cap = cv.VideoCapture('C:/Users/Roman/Documents/Python/Project/Data/image_video_data/Vi.avi')
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = 720
width = 1280
region_of_interest_vertices = [
    (0, height / 2),
    (width / 2, 0),
    (width, height / 2)
]
cut_coeff = [int(height * 0.4), height, 0, width]
# cut_coeff = [int(height * 0.5), int(height * 0.9), int(width * 0.1), int(width * 0.9)]
# test_frame = np.zeros((height, width))
# test_frame = test_frame[cut_coeff[0]:cut_coeff[1], cut_coeff[2]:cut_coeff[3]]
# cut_frame_shape = (test_frame.shape[1], test_frame.shape[0])
# cut_frame_ratio = test_frame.shape[1] / test_frame.shape[0]

# frame_per_sec = 30
# start_frame_count = int(frame_per_sec * 60 * 0)
# stop_frame_count = int(frame_per_sec * 60 * 0.26)
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('C:/Users/Roman/Documents/Python/Project/Data/image_video_data/res1.avi', fourcc, 20.0, (1280, 720))
# frameCount = 0
# fps = cap.get(cv.CAP_PROP_FPS)

initial_trackbar_values = [1, 255, 7]
Utils.initialize_trackbars(initial_trackbar_values)
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # sharpen
# kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])  # edge detection
count = 0
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == 27:
        break
    if ret:
        val_trackbars = Utils.val_trackbars()
        frame = cv.resize(frame, (1280, 720))
        cut_frame = frame[cut_coeff[0]:cut_coeff[1], cut_coeff[2]:cut_coeff[3]]
        test_frame = np.zeros((cut_frame.shape[0], cut_frame.shape[1], 3))
        grey_frame = cv.cvtColor(cut_frame, cv.COLOR_BGR2GRAY)
        grey_copy = grey_frame.copy()
        filtered = Utils.grey_filter(grey_frame)
        contours, hierarchy = cv.findContours(filtered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.2 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            perimeter = cv.arcLength(approx, False)
            if 15 < perimeter < 800:
                cv.drawContours(test_frame, [approx], 0, (0, 255, 0), 2)
        current_time = time.time()
        # cv.putText(color, 'fps: ' + str(int(1 / (current_time - start_time))), (10, 50),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2, cv.LINE_AA)
        cv.imshow('video_frame', test_frame)
        # frameCount += 1
        # if start_frame_count <= frameCount < stop_frame_count:
        #     out.write(result_frame)
        # elif frameCount == stop_frame_count:
        #     break
    else:
        break
cap.release()
cv.destroyAllWindows()
