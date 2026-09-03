import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# the size that the MobileNetV2 trained on ImageNet used
IMAGE_SIZE = (224, 224)
MINIBATCH_SIZE = 32

# create the image datasets
dataset_path = "../dataset"
training_set = image_dataset_from_directory(dataset_path,
                                            validation_split=0.2,
                                            subset="training",
                                            shuffle=True,
                                            image_size=IMAGE_SIZE,
                                            batch_size=MINIBATCH_SIZE,
                                            seed=42)
dev_set = image_dataset_from_directory(dataset_path,
                                       validation_split=0.2,
                                       subset="validation",
                                       shuffle=True,
                                       image_size=IMAGE_SIZE,
                                       batch_size=MINIBATCH_SIZE,
                                       seed=42)

# check the data
# print(training_set.element_spec)
# print(dev_set.element_spec)
# print()
# for images, labels in training_set.take(1):
#     # print(images.shape)
#     # print(labels)
#     plt.figure(figsize=(10, 10))
#
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(f"Label: {labels[i].numpy()}")
#         plt.axis("off")
#
#     plt.show()



