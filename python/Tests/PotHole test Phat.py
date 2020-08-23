import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob
import numpy as np

# BINARY CLASSIFICATION
start_time = time.time()
eval_path = 'C:/Users/Roman/Documents/Python/Project/Data/no_holes/'
model_path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
modelName = 'MobileNetV2Hole_128x128_12512_Detection_Chance=[0.9586]_2020-06-02 22_23_20.922255'
IMAGE_SIZE = [int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0])]
model = tf.keras.models.load_model(model_path + modelName)
non_holes_files = glob(eval_path + '1/*')
chance_threshold = 0.9

prediction_non_holes = []
for e in non_holes_files:
    Img = load_img(e)
    rImg = Img.resize(IMAGE_SIZE)
    Img_array = img_to_array(rImg)
    Img_array = Img_array.reshape((1,) + Img_array.shape)  # Converting into 4 dimension array
    prediction_non_holes.append(Img_array)

# predict on whole evaluation set
prediction_results = []

# 1 - Non-Hole
misclassified = 0
for p in prediction_non_holes:
    predictions = model.predict(p)
    if predictions[0][0] > chance_threshold:
        misclassified += 1
prediction_results.append(1 - misclassified / len(prediction_non_holes))

print("Negative Detection Chance: ", prediction_results[0].__round__(14))
print("--- %s seconds ---" % (time.time() - start_time))
