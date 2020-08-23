import time
import numpy as np
import cv2.cv2 as cv
from PotHole_Detector import UtilsM

stream = 'C:/Users/Roman/Documents/Python/Project/Data/image_video_data/Vi.avi'
cap = cv.VideoCapture(stream)
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
region_of_interest_vertices = [
    (100, height),
    (width / 2, height / 2),
    (width - 100, height)
]

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == 27:
        break
    if ret:
        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grey_copy = grey_frame.copy()
        canny_image = cv.Canny(grey_frame, 100, 150)
        cropped_frame = UtilsM.region_of_interest(canny_image,
                                                  np.array([region_of_interest_vertices], np.int32), )
        roi_frame = UtilsM.find_road_area(cropped_frame, frame, grey_copy)
        frame_copy = roi_frame[1].copy()
        result_frame = UtilsM.detect_holes(roi_frame[0], roi_frame[1], frame_copy)
        current_time = time.time()
        cv.putText(result_frame, 'fps: ' + str(int(1 / (current_time - start_time))), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2, cv.LINE_AA)
        cv.imshow('video_frame', result_frame)
    else:
        break
cap.release()
cv.destroyAllWindows()
