import tensorflow as tf
modelName = 'ResNet50V2_128x128_1780_Detection_Chance=[0.8901]_2020-06-03 12_55_18.130516'
path = 'C:/Users/Roman/Documents/Python/Project/Data/models/'
new_model = tf.keras.models.load_model(path + modelName)

# Create a converter object
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)

# Convert the model
tflite_model = converter.convert()

# Save to file
with open(path + modelName + '.tflite', "wb") as f:
    f.write(tflite_model)
