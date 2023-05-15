import pdb

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    LeakyReLU,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model

if __name__ == "__main__":
    inp = Input([512, 512, 3])
    backbone = VGG16(weights="imagenet", include_top=False, input_tensor=inp)
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in backbone.layers:
        if layer.name.find("pool") != -1:
            features.append(backbone.get_layer(layer.name).output)
    features = features[::-1]
    backbone_func = Model(inputs=inp, outputs=features)

    print(features, backbone_func)
    # pdb.set_trace()
