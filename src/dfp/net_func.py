import pdb

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Add,
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

    features_room_boundary = []
    rbdims = [256, 128, 64, 32]
    x = features[0]
    for i in range(len(rbdims)):
        x = Conv2DTranspose(rbdims[i], 4, strides=2, padding="same")(x)
        xf = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            features[i + 1]
        )
        x = Add()([x, xf])
        x = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(x)
        x = ReLU()(x)
        features_room_boundary.append(x)
    x = Conv2D(3, 1, strides=1, padding="same", dilation_rate=1)(x)
    logits_cw = tf.keras.backend.resize_images(x, 2, 2, "channels_last")
    print(logits_cw)

    print(features_room_boundary)
    # pdb.set_trace()
