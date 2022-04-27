import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

#xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
xs = np.arange(0.0, 150.0, 0.1)

ys = [x + 6.0 for x in xs]
ys = tf.convert_to_tensor(ys)


model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

model.fit(x=xs, y=ys, epochs=200)

print('Prediction is: ', model.predict([10.0]))
