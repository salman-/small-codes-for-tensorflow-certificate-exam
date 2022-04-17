import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

dt = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dt['Origin'] = dt['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
# Change the Origin column to ONE_HOT
dt = pd.get_dummies(dt, columns=['Origin'], prefix='Origin_')

dt.dropna(inplace=True)

dt_single_varibable = dt.copy()
single_variable_train_dt = dt_single_varibable.loc[:, ['Horsepower', 'MPG']].sample(frac=0.8, random_state=0)
single_variable_test_dt = dt_single_varibable.drop(single_variable_train_dt.index)

train_x = single_variable_train_dt['Horsepower']
train_y = single_variable_train_dt['MPG']

test_x = single_variable_test_dt['Horsepower']
test_y = single_variable_test_dt['MPG']

normalization_layer = Normalization(axis=None)
normalization_layer.adapt(single_variable_test_dt['Horsepower'].to_numpy())

model = Sequential()
model.add(Input(shape=(1,)))
model.add(normalization_layer)
model.add(Dense(1))

model.compile(loss='mae', optimizer='sgd', metrics=['mape', 'mae'])
history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), verbose=0, epochs=100)

hist = pd.DataFrame(history.history)
hist.tail()

# ----------------------- PREDICT
print(model.predict(test_x.iloc[:5].to_numpy()),'  ',test_y.to_numpy()[:5])
