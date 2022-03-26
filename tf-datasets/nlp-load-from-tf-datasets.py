import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import TextVectorization,Input,Dense,LSTM,Flatten,GlobalAveragePooling1D,Embedding,Dropout
import numpy as np
import json
import pandas as pd

(train_dt,test_dt),dt_info = tfds.load('imdb_reviews',split=['train','test'],as_supervised=True,with_info=True,shuffle_files=True)

train_sentence = []
test_sentence = []

train_label = []
test_label = []

def load_data(dt,sentences=[],labels=[]):
  for text,label in dt:
    sentences.append(text)
    labels.append(label)
  return sentences,labels

train_sentence,train_label = load_data(train_dt)
test_sentence,test_label = load_data(train_dt)


# Convert list to Tensor
train_sentence = tf.convert_to_tensor(train_sentence)
train_label = tf.convert_to_tensor(train_label)

test_sentence = tf.convert_to_tensor(test_sentence)
test_label = tf.convert_to_tensor(test_label)

text_vectorization_layer = TextVectorization(max_tokens=10000,output_sequence_length=120)
text_vectorization_layer.adapt(train_sentence)

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = Embedding(input_dim=10000,output_dim=128)(x)
x = LSTM(64,return_sequences=True)(x)
x = LSTM(64,return_sequences=True)(x)
x = LSTM(64,return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(64)(x)
x = Flatten()(x)
output = Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(input,output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=train_sentence,y=train_label,validation_data=(test_sentence,test_label),epochs=10)