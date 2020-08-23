import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from sklearn.metrics import confusion_matrix
import itertools


# Makes plot of images array
def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=8, rotation=30)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # """
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


modelName = 'MobileNetV2_224x224_2860_Hole=[0.94736842105263]_Non-Hole=[0.99122807017544]'
test_path = 'C:/Users/Roman/Documents/Python/Project/Data/PotHole Dataset/models/'
new_model = tf.keras.models.load_model(test_path + modelName)
# Check its architecture
# new_model.summary()

# Single Image Prediction
IMAGE_SIZE = [int(modelName.split('_')[1].split('x')[1]), int(modelName.split('_')[1].split('x')[0])]
Img = load_img('C:/Users/Roman/Documents/Python/Project/Data/PotHole Dataset/eval/2 hole/930.JPG')
rImg = Img.resize(IMAGE_SIZE)
Img_array = img_to_array(rImg)
Img_array = Img_array.reshape((1,) + Img_array.shape)  # Converting into 4 dimension array
print(Img_array.shape)
predictions = new_model.predict(Img_array)
print('predictions shape : ', predictions.shape)
print(predictions[0][0], predictions[0][1])
print(round(predictions[0][0], 2), round(predictions[0][1], 2))
breakpoint()

# Multi Image Prediction
IMAGE_SIZE = [int(modelName.split('_')[2].split('x')[0]), int(modelName.split('_')[2].split('x')[1])]
batch_size = 64
test_path = 'C:/Users/Roman/Documents/Python/Project/Data/PotHole Dataset/eval'
test_batches = ImageDataGenerator().flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    classes=['1 not a hole', '2 hole'],
)

# Predict batch
predictions = new_model.predict(test_batches, steps=1)
# print('predictions shape : ', predictions.shape)
# print('predictions: ', predictions)

# Round the predictions
for i in range(0, len(predictions)):
    predictions[i][0] = round(predictions[i][0], 0)
    predictions[i][1] = round(predictions[i][1], 0)
print("Rounded predictions: ", predictions)

# # Show first batch of images
images, labels = test_batches[0]
labels = labels[:, 0]
test_labels = np.array(len(labels))
for i in range(0, len(labels)):
    if labels[i] == 0:
        test_labels = np.append(test_labels, ['Hole ' + str(i)])
    else:
        test_labels = np.append(test_labels, ['Non-Hole ' + str(i)])
plots(images, titles=test_labels[1:])
plt.show()

# # Show confusion matrix
cm = confusion_matrix(labels, predictions[:, 0])
cm_plot_labels = ['hole', 'not a hole']
plot_confusion_matrix(cm, cm_plot_labels)
plt.show()

# # Show some misclassified examples
# misclassified_idx = np.where(predictions[:, 0] != labels)[0]
# print("Misclassified list: ", misclassified_idx, type(misclassified_idx))
# for i in misclassified_idx:
#     print(images[i].shape)
#     plt.imshow(images[i].astype('uint8'))
#     plt.title("True label: %s Predicted: %s" % (labels[i], predictions[:, 0][i]));
#     plt.show();

# Show some misclassified examples v2 (binary)
misclassified_idx = np.where(predictions[:, 0] != labels)[0]
if len(misclassified_idx) != 0:
    print("Misclassified numbers: ", misclassified_idx)
    Miss_imgs = np.zeros((len(misclassified_idx), images.shape[1], images.shape[2], images.shape[3]))
    Miss_labels = np.array(len(misclassified_idx))
    k = 0
    for i in misclassified_idx:
        Miss_imgs[k] = images[i]
        if labels[i] == 0:
            Miss_labels = np.append(Miss_labels, ['Missed Hole ' + str(i)])
        else:
            Miss_labels = np.append(Miss_labels, ['Missed Non-Hole ' + str(i)])
        k += 1
    plots(Miss_imgs, titles=Miss_labels[1:])
    plt.title = 'Wrong Binary Predictions'
    plt.show()
