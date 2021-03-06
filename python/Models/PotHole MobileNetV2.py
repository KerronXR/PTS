import os
import time
import gc
import pandas as pd
import numpy as np
from glob import glob
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Should be before TF import to work
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as PretrainedModel, \
    preprocess_input

model_type = "MobileNetV2"
start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = 'C:/Users/Roman/Documents/Python/Project/Data/mainSet/train'
test_path = 'C:/Users/Roman/Documents/Python/Project/Data/mainSet/test'
eval_path = 'C:/Users/Roman/Documents/Python/Project/Data/mainSet/eval'
model_path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'

IMAGE_SIZE = [128, 128]
batch_size = 32
folders = glob(train_path + '/*[!.txt]')
class_no = len(folders)  # number of classes

train_image_files = glob(train_path + '/*/*')
test_image_files = glob(test_path + '/*/*')
eval_image_files = glob(eval_path + '/*/*')
# train_image_files.sort()
# test_image_files.sort()
# eval_image_files.sort()

ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False,
    classes=class_no
)

train_steps = int(np.ceil(len(train_image_files) / batch_size))
validation_steps = int(np.ceil(len(test_image_files) / batch_size))
eval_steps = int(np.ceil(len(eval_image_files) / batch_size))

# Keras image data generator returns classes one-hot encoded
# create an instance of ImageDataGenerator
gen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

gen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
gen_eval = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = gen_train.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)
test_generator = gen_test.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)

# create a 2nd train generator which does not use data augmentation
# to get the true train accuracy
eval_generator = gen_eval.flow_from_directory(
    eval_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)

# @off
optimizer_list = [
    # tf.keras.optimizers.SGD,       # 0
    # tf.keras.optimizers.RMSprop,   # 1
    tf.keras.optimizers.Adagrad,   # 2
    tf.keras.optimizers.Adadelta,  # 3
    tf.keras.optimizers.Adam,      # 4
    tf.keras.optimizers.Adamax,    # 5
    tf.keras.optimizers.Nadam      # 6
]

optimizer_rate = [
    # 1               # 0
    # 0.1             # 1
    0.01,           # 2
    0.001,          # 3
    0.0001,         # 4
    0.00001,        # 5
    0.000001        # 6
]

loss_functions_list = [
    'mean_squared_error',               # 0
    'mean_absolute_error',              # 1
    'mean_absolute_percentage_error',   # 2
    'mean_squared_logarithmic_error',   # 3
    'squared_hinge',                    # 4
    'hinge',                            # 5
    'categorical_hinge',                # 6
    'logcosh',                          # 7
    'huber_loss',                       # 8
    'categorical_crossentropy',         # 9
    'sparse_categorical_crossentropy',  # 10
    'binary_crossentropy',              # 11
    'kullback_leibler_divergence',      # 12
    'poisson',                          # 13
    'cosine_proximity',                 # 14
    'is_categorical_crossentropy'       # 15
]

metrics_list = [
    'accuracy',                            # 0
    'binary_accuracy',                     # 1
    'categorical_accuracy',                # 2
    'sparse_categorical_accuracy',         # 3
    'top_k_categorical_accuracy',          # 4
    'sparse_top_k_categorical_accuracy',   # 5
    'cosine_proximity'                     # 6
]

# train loop routine:                  # default:
cut = 0  # --------------------------- 0 - only dense layer cut
layers_to_train = 1  # --------------- 1 - only new dense layer trained
epochs = 5  # ------------------------ 5 epochs
loss = 9  # -------------------------- 9 - categorical_crossentropy
opt = 4  # --------------------------- 4 - adam
opt_rate = 3  # ---------------------- 1 - 0.001
metrics = 0  # ----------------------- 0 - accuracy

for cut in range(0, 1):  # ----------------------------------------------------- number of top layers cut
    # for layers_to_train in range(1, 31):  # ------------------------------------ number of top layers to train
    for epochs in range(6, 15):  # ------------------------------------------ number of epochs
        #         for loss in range(0, len(loss_functions_list)):  # ----------------- loss functions
        for opt in range(0, len(optimizer_list)):  # ------------------- optimizers
            for opt_rate in range(0, len(optimizer_rate)):  # ---------- optimizer rates
                for j in range(0, 10):  # ------------------------------- iterations
                    model_start_time = time.time()
                    x = Flatten()(ptm.layers[-cut].output)
                    x = Dense(class_no, activation='softmax')(x)

                    # create a model object
                    model = Model(inputs=ptm.input, outputs=x)

                    # reset trainable layers
                    for layer in model.layers:
                        layer.trainable = True

                    # freeze pretrained model weights up to specific value
                    for layer in model.layers[:-layers_to_train]:
                        layer.trainable = False

                    model.compile(
                        loss=loss_functions_list[loss],
                        optimizer=optimizer_list[opt](lr=optimizer_rate[opt_rate]),
                        metrics=(metrics_list[metrics])
                    )

                    model.fit(
                        train_generator,
                        validation_data=test_generator,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        validation_steps=validation_steps,
                    )

                    eval_val = model.evaluate(
                        eval_generator,
                        steps=eval_steps
                    )

                    print(eval_val)
                    hyper_parameters = (model_type + '__' +
                                        'ImageSize=' + str(IMAGE_SIZE[0]) + 'x' + str(IMAGE_SIZE[1]) + '__' +
                                        'FileNumber=' + str(len(train_image_files)) + '__' +
                                        'Eval=' + str(eval_val) + '__' +
                                        'BatchSize=' + str(batch_size) + '__' +
                                        'Epochs=' + str(epochs) + '__' +
                                        'Optimizer=' + str(type(model.optimizer).__name__) + '__' +
                                        'Opt.Rate=' + str(optimizer_rate[opt_rate]) + '__' +
                                        'Loss=' + str(loss_functions_list[loss]) + '__' +
                                        'Metrics=' + metrics_list[metrics] + '__' +
                                        'Cut=' + str(cut) + '__' +
                                        'Layers=' + str(layers_to_train))

                    if eval_val[0] < 0.2:  # if loss is not bigger than 0.2

                        model_name = (model_type + '_' + str(IMAGE_SIZE[0]) + 'x' + str(IMAGE_SIZE[1]) + '_' +
                                      str(len(train_image_files)) + '_' + 'Loss=[' +
                                      str(eval_val[0].__round__(4)) + ']_' +
                                      str(datetime.datetime.now()).replace(':', '_'))

                        model.save(model_path + model_name)
                        result_file = open(model_path + model_name + '/HyperParameters.txt', 'w')
                        for h in hyper_parameters.split('__'):
                            result_file.write(h + '\n')
                        result_file.close()
                    print(hyper_parameters)
                    print("--- %s seconds ---" % (time.time() - model_start_time))
                    tf.keras.backend.clear_session()
                    gc.collect()
print("--- %s seconds ---" % (time.time() - start_time))
