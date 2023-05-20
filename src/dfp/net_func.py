import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
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


def resnet50_backbone(x, feature_names):
    backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    backbone = Model(
        inputs=x, outputs=backbone.get_layer(feature_names[-1]).output
    )
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in feature_names:
        features.append(backbone.get_layer(layer).output)
    features = features[::-1]
    return features


def mobilenet_backbone(x, feature_names):
    backbone = MobileNet(weights="imagenet", include_top=False, input_tensor=x)
    backbone = Model(
        inputs=x, outputs=backbone.get_layer(feature_names[-1]).output
    )
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in feature_names:
        features.append(backbone.get_layer(layer).output)
    features = features[::-1]
    return features


def mobilenetv2_backbone(x, feature_names):
    backbone = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=x
    )
    backbone = Model(
        inputs=x, outputs=backbone.get_layer(feature_names[-1]).output
    )
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in feature_names:
        features.append(backbone.get_layer(layer).output)
    features = features[::-1]
    return features


def vgg16_backbone(x, feature_names):
    backbone = VGG16(weights="imagenet", include_top=False, input_tensor=x)
    backbone = Model(
        inputs=x, outputs=backbone.get_layer(feature_names[-1]).output
    )
    for layer in backbone.layers:
        layer.trainable = False

    features = []
    for layer in feature_names:
        features.append(backbone.get_layer(layer).output)
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


def deepfloorplanFunc(config: argparse.Namespace = None):
    inp = Input([512, 512, 3])
    if config is None:
        rbdims = [256, 128, 64, 32]
        features = vgg16_backbone(
            inp,
            [
                "block1_pool",
                "block2_pool",
                "block3_pool",
                "block4_pool",
                "block5_pool",
            ],
        )
    elif config is not None:
        rbdims = config.feature_channels
        if config.backbone == "resnet50":
            features = resnet50_backbone(inp, config.feature_names)
        elif config.backbone == "vgg16":
            features = vgg16_backbone(inp, config.feature_names)
        elif config.backbone == "mobilenetv1":
            features = mobilenet_backbone(inp, config.feature_names)
        elif config.backbone == "mobilenetv2":
            features = mobilenetv2_backbone(inp, config.feature_names)
    assert len(features) == 5, "Not enough 5 features..."

    features_room_boundary = []
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
