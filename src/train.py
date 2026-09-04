import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from pprint import pprint

import matplotlib.pyplot as plt

from model import cat_classifier_model

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

MODEL_PATH = "../models/cat_image_classifier.keras"
WEIGHT_PATH = "../models/weights/cat_image_classifier.weights.h5"
model = cat_classifier_model(IMAGE_SIZE)
# transfer learning stage
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(training_set, epochs=5, validation_data=dev_set)

# print out the history
pprint(history.history)

'''
save the weights

Training accuracy - 98%
dev accuracy - 96%
'''
model.save(MODEL_PATH)
model.save_weights(WEIGHT_PATH)
