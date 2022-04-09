import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Lambda

series = list(np.arange(10))

WINDOW_SIZE = 5
BATCH_SIZE = 5
BUFFER_SIZE = 1000

print(series)


def create_windowed_dt(series, WINDOW_SIZE, BATCH_SIZE, BUFFER_SIZE):
    dt = tf.data.Dataset.from_tensor_slices(series)
    dt = dt.window(WINDOW_SIZE, shift=1, drop_remainder=True)
    dt = dt.flat_map(Lambda(lambda window: window.batch(BATCH_SIZE)))
    dt = dt.map(Lambda(lambda window: (window[:-1], window[-1:])))
    dt = dt.shuffle(buffer_size=BUFFER_SIZE)
    return dt.batch(BATCH_SIZE).prefetch(1)

dt = create_windowed_dt(series, WINDOW_SIZE, BATCH_SIZE, BUFFER_SIZE)

for window in dt:
    print(window[0].numpy(), ' ', window[1].numpy())
