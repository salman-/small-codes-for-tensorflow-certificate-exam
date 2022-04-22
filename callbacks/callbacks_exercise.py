import zipfile
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0, resnet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import pandas as pd

import zipfile

# Download zip file of pizza_steak images
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("pizza_steak.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_dir = './pizza_steak/train/'
test_dir = './pizza_steak/test/'
IMAGE_SIZE = (224, 224)

# data_generator = ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,rotation_range=0.2,width_shift_range=0.2)
data_generator = ImageDataGenerator(rescale=1. / 255)

train_dt = data_generator.flow_from_directory(directory=train_dir, batch_size=32, target_size=IMAGE_SIZE,
                                              class_mode='binary')
test_dt = data_generator.flow_from_directory(directory=test_dir, batch_size=32, target_size=IMAGE_SIZE,
                                             class_mode='binary')

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(224, 224, 3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding="valid"))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

custom_callbacks = [EarlyStopping(patience=5, monitor='val_loss'),
                    ModelCheckpoint('./model.h5', save_best_only=True, monitor='val_loss'),
                    LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20)),
                    ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.0001)]

model.fit(train_dt, validation_data=test_dt, callbacks=custom_callbacks, epochs=10)
