import os
import time
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import sys
from glob import glob

modelType = "CNN"
start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = 'C:/Users/Roman/Documents/Python/Project/Data/train'
test_path = 'C:/Users/Roman/Documents/Python/Project/Data/test'
eval_path = 'C:/Users/Roman/Documents/Python/Project/Data/eval'
model_path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'

# These images are pretty big and of different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [256, 128]
batch_size = 8
epochs = 15

# useful for getting number of files
train_image_files = glob(train_path + '/*/*.jpg')
test_image_files = glob(test_path + '/*/*.jpg')
eval_image_files = glob(eval_path + '/*/*.jpg')

# useful for getting number of classes
folders = glob(train_path + '/*')
print(folders)

# number of classes
K = len(folders)
print("number of classes:", K)

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         shuffle=True,
                                                         target_size=IMAGE_SIZE,
                                                         batch_size=batch_size,
                                                         )

test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                        target_size=IMAGE_SIZE,
                                                        batch_size=batch_size,
                                                        )

eval_batches = ImageDataGenerator().flow_from_directory(eval_path,
                                                        target_size=IMAGE_SIZE,
                                                        batch_size=batch_size,
                                                        )
# Build the model using the functional API
i = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
print(i)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit
r = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=epochs,
    steps_per_epoch=int(np.ceil(len(train_image_files) / batch_size)),
    validation_steps=int(np.ceil(len(test_image_files) / batch_size)),
)

eval_val = model.evaluate(
    eval_batches,
    steps=int(np.ceil(len(eval_image_files) / batch_size)))

# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# model.summary()

print(eval_val)
model.save('C:/Users/Roman/Documents/Python/Project/Data/PotHole Dataset/models/Pothole_Model_' +
           str(IMAGE_SIZE[0]) + 'x' + str(IMAGE_SIZE[1]) + '_' +
           'Batch=' + str(batch_size) +
           '_EP=' + str(epochs) + '_' + str(len(train_image_files)) + '_' + str(eval_val) + '_' + modelType)
print("--- %s seconds ---" % (time.time() - start_time))
