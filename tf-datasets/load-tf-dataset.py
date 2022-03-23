import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(train_data, test_data),ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, download=False,
                                    shuffle_files=True,with_info=True)

print(ds_info.features)
print(ds_info.features['label'].names)


sample_data = train_data.take(1)
for image,label in sample_data:
    print("image type:",type(image)," label type:",type(label))
    print("image shape:",image.shape," label shape:",label.shape)

plt.imshow(image)
plt.show()