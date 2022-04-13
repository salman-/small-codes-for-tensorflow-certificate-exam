import zipfile
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0, resnet50
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

# !wget !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip
zip_ref = zipfile.ZipFile("pizza_steak.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_directory = './pizza_steak/train/'
test_directory = './pizza_steak/test/'
IMAGE_SIZE = (224, 224)

image_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          zoom_range=0.2,
                                          shear_range=0.2,
                                          rotation_range=0.2)

train_dt = image_data_generator.flow_from_directory(directory=train_directory,
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    target_size=IMAGE_SIZE)

test_dt = image_data_generator.flow_from_directory(directory=test_directory,
                                                   class_mode='categorical',
                                                   batch_size=32,
                                                   target_size=IMAGE_SIZE)

# Get the class names
# test_dt.class_indices

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dt,
          epochs=5,
          validation_data=test_dt,
          validation_steps=len(test_dt))


# --------------------- PREDICTION ---------------------------

def load_image_for_prediction(img_path):
    img = tf.io.read_file(img_path)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size=IMAGE_SIZE)

    # Rescale the image (get all values between 0 and 1)
    img = img / 255.
    return tf.expand_dims(img, axis=0)


# ----------------- Get categories name
class_names = [x for x in test_dt.class_indices.keys()]
class_names

# ------------------------- Get prediction probability
img_path = './pizza_steak/test/pizza/1001116.jpg'  # it is a pizza image
model.predict(load_image_for_prediction(img_path))

img_path = './pizza_steak/test/steak/1064847.jpg'  # it is a steak image
model.predict(load_image_for_prediction(img_path))

#------------ Get most probable class
prediction_probabilities = [0.08698346, 0.9142322 ]
max_values = np.max(prediction_probabilities)
class_names[prediction_probabilities.index(max_values)]