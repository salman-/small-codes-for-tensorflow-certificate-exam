import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import pandas as pd

(train_data, test_data), ds_info = tfds.load(name='mnist', split=['train', 'test'], as_supervised=True, with_info=True)
print(len(ds_info.features['label'].names))
print(ds_info.features)


def pre_process_image(image, label, new_size=[224, 224]):
    image = tf.cast(image, tf.float64)
    label = tf.cast(label, tf.float64)
    image = tf.image.resize(image, new_size)
    return image, label


train_data = train_data.map(pre_process_image)
train_data = train_data.shuffle(buffer_size=1000)
train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(pre_process_image)
test_data = test_data.shuffle(buffer_size=1000)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

model = Sequential()
model.add(Input(shape=(224, 224, 1)))
model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(len(ds_info.features['label'].names), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plot_model(model, "./model-architecture-picture/image-classifier.png", show_shapes=True)

modelCheckpoint = ModelCheckpoint('./model/image-classifier.h5', monitor='val_loss', save_best_only=True)
result = model.fit(train_data, validation_data=test_data, epochs=6, callbacks=modelCheckpoint)

print(result.history)

dt = pd.DataFrame(result.history)
dt.plot()