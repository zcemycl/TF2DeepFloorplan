from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    Multiply,
    ReLU,
)
from tensorflow.keras.models import Model


def mobilenet_backbone(x):
    layer_names = [
        "conv_pw_1_relu",  # 256x256x64
        "conv_pw_3_relu",  # 128x128x128
        "conv_pw_5_relu",  # 64x64x256
        "conv_pw_7_relu",  # 32x32x512
        "conv_pw_13_relu",  # 16x16x1280
    ]
    backbone = MobileNet(weights="imagenet", include_top=False, input_tensor=x)
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in backbone.layers:
        if layer.name in layer_names:
            features.append(backbone.get_layer(layer.name).output)
    features = features[::-1]
    return features


def mobilenetv2_backbone(x):
    layer_names = [
        "block_1_expand_relu",  # 256x256x96
        "block_3_expand_relu",  # 128x128x144
        "block_5_expand_relu",  # 64x64x192
        "block_13_expand_relu",  # 32x32x576
        "out_relu",  # 16x16x1280
    ]
    backbone = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=x
    )
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in backbone.layers:
        if layer.name in layer_names:
            features.append(backbone.get_layer(layer.name).output)
    features = features[::-1]
    return features


def vgg16_backbone(x):
    backbone = VGG16(weights="imagenet", include_top=False, input_tensor=x)
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in backbone.layers:
        if layer.name.find("pool") != -1:
            features.append(backbone.get_layer(layer.name).output)
    features = features[::-1]
    return features


def vertical_horizontal_filters(
    h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    return np.ones([1, h, 1, 1]), np.ones([w, 1, 1, 1])


def diagonal_filters(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    d = np.eye(h, w)
    dr = np.eye(h, w)
    dr = dr.reshape([h, w, 1])
    dr = np.flip(dr, 1)
    return d.reshape((h, w, 1, 1)), dr.reshape((h, w, 1, 1))


def non_local_context(xf, x_, rbdim, stride=4):
    config_non_trainable = {
        "filters": 1,
        "trainable": False,
        "padding": "same",
        "use_bias": False,
    }

    _, H, W, _ = xf.shape

    hs = H // stride if (H // stride) > 1 else (stride - 1)
    vs = W // stride if (W // stride) > 1 else (stride - 1)
    hs = hs if (hs % 2 != 0) else hs + 1
    vs = hs if (vs % 2 != 0) else vs + 1
    vf, hf = vertical_horizontal_filters(vs, hs)
    df, dfr = diagonal_filters(vs, hs)

    v = Conv2D(
        kernel_size=[1, vs],
        **config_non_trainable,
        weights=[vf],
    )(x_)
    h = Conv2D(
        kernel_size=[hs, 1],
        **config_non_trainable,
        weights=[hf],
    )(x_)
    d = Conv2D(
        kernel_size=[hs, vs],
        **config_non_trainable,
        weights=[df],
    )(x_)
    dr = Conv2D(
        kernel_size=[hs, vs],
        **config_non_trainable,
        weights=[dfr],
    )(x_)
    c = Add()([v, h, d, dr])
    c = Multiply()([xf, c])
    c = Conv2D(rbdim, 1, strides=1, padding="same", dilation_rate=1)(c)
    c = Concatenate(axis=3)([x_, c])
    x = Conv2D(rbdim, 3, strides=1, padding="same", dilation_rate=1)(c)
    return x


def attention(xf, x_, rbdim):
    xf = Conv2D(rbdim, 3, strides=1, padding="same", dilation_rate=1)(xf)
    xf = ReLU()(xf)
    xf = Conv2D(rbdim, 3, strides=1, padding="same", dilation_rate=1)(xf)
    xf = ReLU()(xf)
    xf = Conv2D(1, 1, strides=1, padding="same", dilation_rate=1)(xf)
    xf = tf.keras.activations.sigmoid(xf)

    x_ = Conv2D(rbdim, 3, strides=1, padding="same", dilation_rate=1)(x_)
    x_ = ReLU()(x_)
    x_ = Conv2D(1, 1, strides=1, padding="same", dilation_rate=1)(x_)
    x_ = Multiply()([xf, x_])

    return non_local_context(xf, x_, rbdim)


def deepfloorplanFunc():

    inp = Input([512, 512, 3])
    # features = vgg16_backbone(inp)
    features = mobilenet_backbone(inp)

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

    x = features[0]

    for i in range(len(rbdims)):
        x = Conv2DTranspose(rbdims[i], 4, strides=2, padding="same")(x)
        xf = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            features[i + 1]
        )
        x = Add()([x, xf])
        x = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(x)
        x = ReLU()(x)

        # attention and contexture
        x = attention(features_room_boundary[i], x, rbdims[i])

    x = Conv2D(9, 1, strides=1, padding="same", dilation_rate=1)(x)
    logits_r = tf.keras.backend.resize_images(x, 2, 2, "channels_last")
    print(logits_cw.shape, logits_r.shape)

    return Model(inputs=inp, outputs=[logits_r, logits_cw])


if __name__ == "__main__":
    model = deepfloorplanFunc()
    print(model.summary())
    for layer in model.layers:
        print(layer.name, layer.trainable)
    model.save("/tmp/tmp")
