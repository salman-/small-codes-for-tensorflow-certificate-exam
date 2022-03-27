import zipfile
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0,resnet50
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd


#!wget https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip
zip_ref = zipfile.ZipFile("101_food_classes_10_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_directory = './101_food_classes_10_percent/train/'
test_directory = './101_food_classes_10_percent/test/'
IMAGE_SIZE = (224,224)

image_data_generator = ImageDataGenerator(rescale=1./255,
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