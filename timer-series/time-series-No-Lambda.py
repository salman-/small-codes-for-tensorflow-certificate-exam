import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Lambda, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd

# Sunspots.csv from Kaggle
#!gdown --id 1bLnqPgwoSh6rHz_DKDdDeQyAyl8_nqT5

dt = pd.read_csv('./Sunspots.csv',parse_dates=["Date"])
dt.rename(columns={'Monthly Mean Total Sunspot Number':'value'},inplace=True)
#dt
split_index = 3000
values = dt['value'].values

train = values[:split_index]
test = values[split_index:]

window_size = 64
batch_size = 32  # Length of period of time to consider as a batch
shuffle_buffer = 1000


def get_windowed_data(series, window_size, batch_size, shuffle_buffer):
    dt = tf.data.Dataset.from_tensor_slices(series)
    dt = dt.window(window_size, shift=1, drop_remainder=True)
    dt = dt.flat_map(lambda window: window.batch(window_size))  # make each window a batch
    dt = dt.map(lambda window: (window[:-1], window[-1:]))  # consider the last element as label and the rest as window
    dt = dt.shuffle(buffer_size=shuffle_buffer)
    return dt.batch(batch_size).prefetch(1)

train_dt = get_windowed_data(train, window_size, batch_size, shuffle_buffer)
test_dt = get_windowed_data(test, window_size, batch_size, shuffle_buffer)

tf.keras.backend.clear_session()

model = Sequential()
#model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(Conv1D( filters=32,kernel_size=5,activation='relu',input_shape=[None,1] ))
model.add(LSTM(60, return_sequences=True))
model.add(LSTM(60, return_sequences=True))

model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mae', optimizer='sgd', metrics=['mae'])

callbacks = [ModelCheckpoint('./model.h5', save_best_only=True, monitor='val_loss')#,ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.0001)
             ]

model.fit(train_dt, validation_data=test_dt, callbacks=callbacks, epochs=100)