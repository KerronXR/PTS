import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as PretrainedModel, \
    preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

# BINARY CLASSIFICATION
start_time = time.time()
eval_path = 'C:/Users/Roman/Documents/Python/Project/Data/no_holes/'
model_path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
modelName = 'MobileX_160x160_Hole=2020-05-28 00_37_48.091996_-095-'
IMAGE_SIZE = [int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0])]
model = tf.keras.models.load_model(model_path + modelName)
non_holes_files = glob(eval_path + '1/*')
chance_threshold = 0.9

ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)

# map the data into feature vectors
x = Flatten()(ptm.output)

# create a model object
model_p = Model(inputs=ptm.input, outputs=x)

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
    feats = model_p.predict(p)
    predictions = model.predict(feats)
    if predictions[0][0] > chance_threshold:
        misclassified += 1
prediction_results.append(1 - misclassified / len(prediction_non_holes))

print("Negative Detection Chance: ", prediction_results[0].__round__(14))
print("--- %s seconds ---" % (time.time() - start_time))
