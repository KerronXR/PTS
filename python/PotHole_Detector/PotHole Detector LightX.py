import time
import numpy as np
import cv2.cv2 as cv
from PotHole_Detector import Utils

cap = cv.VideoCapture('C:/Users/Roman/Documents/Python/Project/Data/image_video_data/Vi.avi')
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
cut_coeff = [int(height * 0.5), int(height * 0.9), int(width * 0.1), int(width * 0.9)]
test_frame = np.zeros((height, width))
test_frame = test_frame[cut_coeff[0]:cut_coeff[1], cut_coeff[2]:cut_coeff[3]]
cut_frame_shape = (test_frame.shape[1], test_frame.shape[0])
cut_frame_ratio = test_frame.shape[1] / test_frame.shape[0]

# frame_per_sec = 30
# start_frame_count = int(frame_per_sec * 60 * 0)
# stop_frame_count = int(frame_per_sec * 60 * 0.26)
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('C:/Users/Roman/Documents/Python/Project/Data/image_video_data/res1.avi', fourcc, 20.0, (1280, 720))
# frameCount = 0
# fps = cap.get(cv.CAP_PROP_FPS)
initial_trackbar_values = [0, 0, 1, 60, 10]
# wT,hT,wB,hB, thresh_val, adapt_val, HT, BlockSize, SubConst
Utils.initialize_trackbars(initial_trackbar_values)
count = 0
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if cv.waitKey(30) & 0xFF == 27:
        break
    if ret:
        val_trackbars = Utils.val_trackbars()
        cut_frame = frame[cut_coeff[0]:cut_coeff[1], cut_coeff[2]:cut_coeff[3]]
        # color = Utils.color_filter(cut_frame)
        grey_frame = cv.cvtColor(cut_frame, cv.COLOR_BGR2GRAY)
        roi_frame = Utils.find_road_area(grey_frame, cut_frame, cut_frame_shape, cut_frame_ratio)
        Utils.detect_holes(roi_frame, frame, cut_frame, val_trackbars, count)
        # count += 2
        current_time = time.time()
        cv.putText(cut_frame, 'fps: ' + str(int(1 / (current_time - start_time))), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2, cv.LINE_AA)
        cv.imshow('video_frame', cut_frame)
        # frameCount += 1
        # if start_frame_count <= frameCount < stop_frame_count:
        #     out.write(frame)
        # elif frameCount == stop_frame_count:
        #     break
    else:
        break
cap.release()
cv.destroyAllWindows()
