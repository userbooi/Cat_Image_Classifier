# supress warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path

# set some constants
MODEL_PATH = "../models/cat_image_classifier.keras"
IMAGE_SIZE = (224, 224)

# load the model
model = load_model(MODEL_PATH)

# initialize some stuff
folder = Path("images")
all_files = list(folder.iterdir())
total = len(all_files)
correct = 0

# loop through all the files
for img_path in all_files:
    # load the image
    img_array = np.array(Image.open(img_path).resize(IMAGE_SIZE).convert("RGB"))
    plt.imshow(img_array)
    # expand dims
    # img_array = np.expand_dims(img_array, axis=0)
    # print(img_array.shape)
    # create the label
    # the probabilities are flipped
    # 0 - cat
    # 1 - non-cat
    label = (0 if "cat" in img_path.name else 1)

    pred = model.predict(img_array[np.newaxis, :, :, :])
    probability = tf.nn.sigmoid(pred).numpy().item()
    # print(probability)

    # round off
    if probability > 0.5:
        probability = 1
        print("its not a cat")
    else:
        probability = 0
        print("its a cat")

    # compare
    correct += (1 if probability == label else 0)

print(f"\nAccuracy: {round(correct / total * 100, 2)}%")
