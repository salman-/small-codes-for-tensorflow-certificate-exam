import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

""" ------------------------------ Visualize the sample entry
plt.figure()
plt.imshow(train_images[0])
plt.show()"""

# ----------------------------------------- Conv2D without Lambda
train_images = train_images / 255.0
train_images = tf.expand_dims(train_images,axis=-1)
test_images = test_images / 255.0
test_images = tf.expand_dims(test_images,axis=-1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(4,kernel_size=3),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, validation_data=(test_images, test_labels), epochs=10)

# ------------------------------------------ Dense layer

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, validation_data=(test_images, test_labels), epochs=10)

# -------------------------- Conv2D Using Lambda layer
model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    tf.keras.layers.Input(shape=(28, 28, 3)),
    tf.keras.layers.Conv2D(4, kernel_size=3),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, validation_data=(test_images, test_labels), epochs=10)
