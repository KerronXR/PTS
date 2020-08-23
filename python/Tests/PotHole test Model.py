import os
import time
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

start_time = time.time()
eval_path = 'C:/Users/Roman/Documents/Python/Project/Data/eval'
model_path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
modelName = 'MobileNetV2_160x160_4500_Hole=[0.93]_Non-Hole=[0.61333333333333]'
eval_folders = glob(eval_path + '/*[!.txt]')
class_no = len(eval_folders)  # number of classes
IMAGE_SIZE = [int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0])]
model = tf.keras.models.load_model(model_path + modelName)

# Prepare prediction testing images
prediction_images = []
for c in range(0, class_no):
    image_files = glob(eval_path + '/' + str(c) + '/*')
    class_batch = []
    for e in image_files:
        Img = load_img(e)
        rImg = Img.resize(IMAGE_SIZE)
        Img_array = img_to_array(rImg)
        Img_array = Img_array.reshape((1,) + Img_array.shape)  # Converting into 4 dimension array
        class_batch.append(Img_array)
    prediction_images.append(class_batch)


# predict on whole evaluation set
# iterating class by class and predicting on each image to see if model misses
prediction_results = []
for i in range(0, class_no):
    misclassified = 0
    for p in range(0, len(prediction_images[i])):
        predictions = model.predict(prediction_images[i][p])
        if predictions[0][i] < 0.5:
            misclassified += 1
    prediction_results.append(1 - (misclassified / len(prediction_images[i])))

predict_sum = 0
for p in prediction_results:
    predict_sum += p
detection_chance = predict_sum / len(prediction_results)

print("Detection Chance: ", detection_chance.__round__(14))
print("Hole Detection Chance: ", prediction_results[0].__round__(14))
print("NoN-Hole Detection Chance: ", prediction_results[1].__round__(14))
print("--- %s seconds ---" % (time.time() - start_time))
