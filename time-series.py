import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Lambda, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Download Bitcoin historical data from GitHub
# Note: you'll need to select "Raw" to download the data in the correct format
#!wget https: // raw.githubusercontent.com / mrdbourke / tensorflow - deep - learning / main / extras / BTC_USD_2013 - 10 - 01_2021 - 05 - 18 - CoinDesk.csv

EPOCHS_NR = 10

# Parse dates and set date column to index
df = pd.read_csv("/content/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=['Date'])  # parse the date column (tell pandas column 1 is a datetime)
df = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={'Closing Price (USD)': 'Price'})
df.head()

timesteps = df.index.to_numpy()
prices = df['Price'].to_numpy()

split_index = round(.8 * len(df))
print(split_index)

train_x = timesteps[split_index:]
train_y = prices[split_index:]

test_x = timesteps[:split_index]
test_y = prices[:split_index]

window_size = 5
batch_size = 30  # Length of period of time to consider as a batch
shuffle_buffer = 1000


def get_windowed_data(series, window_size, batch_size, shuffle_buffer):
    dt = tf.data.Dataset.from_tensor_slices(prices)
    dt = dt.window(window_size, shift=1, drop_remainder=True)
    dt = dt.flat_map(lambda window: window.batch(window_size))  # make each window a batch
    dt = dt.map(lambda window: (window[:-1], window[-1:]))  # consider the last element as label and the rest as window
    dt = dt.shuffle(buffer_size=shuffle_buffer)
    return dt.batch(batch_size).prefetch(1)


train_dt = get_windowed_data(train_y, window_size, batch_size, shuffle_buffer)
test_dt = get_windowed_data(test_y, window_size, batch_size, shuffle_buffer)

# for window_dt in train_dt:
#    print(window_dt[0].numpy()," ",window_dt[1].numpy())

# Dense layer
model = Sequential()
model.add(Input(window_size - 1))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mape'])
model.fit(train_dt, validation_data=test_dt, epochs=EPOCHS_NR)

# RNN layer

model = Sequential()
model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(SimpleRNN(40, return_sequences=True))
model.add(SimpleRNN(40))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mape'])
model.fit(train_dt, epochs=EPOCHS_NR)

# Bidirectional RNN layer

model = Sequential()
model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(Bidirectional(SimpleRNN(40, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(40)))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mape'])
model.fit(train_dt, validation_data=test_dt, epochs=EPOCHS_NR)

# Conv1D layer

model = Sequential()
model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dense(1))

model.compile(loss='huber', optimizer='adam', metrics=['mae', 'mape'])
model.fit(train_dt, validation_data=(test_dt), epochs=EPOCHS_NR)

# LSTM layer

model = Sequential()
model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
model.add(LSTM(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='huber', optimizer='adam', metrics=['mae', 'mape'])
model.fit(train_dt, validation_data=(test_dt), epochs=EPOCHS_NR)
