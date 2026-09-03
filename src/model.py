import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, GlobalAveragePool2D, RandomFlip, RandomRotation, \
    RandomZoom, RandomContrast, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model

# create the data augmenter that returns a sequential model to augment the existing dataset to create more images
def data_augmenter():
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomZoom(0.1),
        RandomContrast(0.1),
        RandomRotation(0.1)
    ])

    return data_augmentation

# create a function to get the preprocessing used my mobile net
def preprocessor():
    return MobileNetV2.preprocess_input

# create the cat classifier
def cat_classifier_model(img_size, data_augmentation=data_augmenter(), preprocessing=preprocessor()):

    # set the input size to the size of an image plus the channels
    X = Input(shape=img_size+(3,))

    # get the pretrain mobile net
    mobile_net = MobileNetV2(
        weights="imagenet",
        include_top=False
    )
    # set its weights as untrainable
    mobile_net.trainable = False

    # augment the data and preprocess
    X = data_augmentation(X)
    # preprocess the data
    X = preprocessing(X)

    # pass it through the trained mobile net
    # set it to inference mode so the batch norm works fine
    X = mobile_net(X, training=False)

    # create the new top layers to complete the transfer learning
    X = GlobalAveragePool2D()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.2)(X)
    # only use linear to use from_logits later
    output = Dense(1, activation='linear')(X)

    model = Model(inputs=X, outputs=output)

    return model
