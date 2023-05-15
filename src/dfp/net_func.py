# import pdb
from typing import Tuple

import tensorflow as tf
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


@tf.function
def vertical_horizontal_filters(h: int, w: int) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.ones([1, h, 1, 1]), tf.ones([w, 1, 1, 1])


@tf.function
def diagonal_filters(h: int, w: int) -> Tuple[tf.Tensor, tf.Tensor]:
    d = tf.eye(h, w)
    dr = tf.eye(h, w)
    dr = tf.reshape(dr, [h, w, 1])
    dr = tf.reverse(dr, [1])
    return tf.reshape(d, (h, w, 1, 1)), tf.reshape(dr, (h, w, 1, 1))


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
    # backbone_func = Model(inputs=inp, outputs=features)

    print(features)

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
    stride = 4
    for i in range(len(rbdims)):
        print("----------------------")
        x = Conv2DTranspose(rbdims[i], 4, strides=2, padding="same")(x)
        xf = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            features[i + 1]
        )
        x = Add()([x, xf])
        x = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(x)
        x = ReLU()(x)

        # attention and contexture
        x_ = x
        xf = features_room_boundary[i]
        print(xf.shape)
        xf = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            xf
        )
        xf = ReLU()(xf)
        xf = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            xf
        )
        xf = ReLU()(xf)
        xf = Conv2D(1, 1, strides=1, padding="same", dilation_rate=1)(xf)
        xf = tf.keras.activations.sigmoid(xf)

        x_ = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(
            x_
        )
        x_ = ReLU()(x_)
        x_ = Conv2D(1, 1, strides=1, padding="same", dilation_rate=1)(x_)
        x_ = Multiply()([xf, x_])

        _, H, W, _ = xf.shape
        print(H, W)
        hs = H // stride if (H // stride) > 1 else (stride - 1)
        vs = W // stride if (W // stride) > 1 else (stride - 1)
        hs = hs if (hs % 2 != 0) else hs + 1
        vs = hs if (vs % 2 != 0) else vs + 1
        print(hs, vs)
        vf, hf = vertical_horizontal_filters(vs, hs)
        print(vf.shape, hf.shape)
        df, dfr = diagonal_filters(vs, hs)
        print(df.shape, dfr.shape)

        v = Conv2D(
            1,
            [1, vs],
            strides=1,
            padding="same",
            trainable=False,
            use_bias=False,
            weights=[vf],
        )(x_)
        h = Conv2D(
            1,
            [hs, 1],
            strides=1,
            padding="same",
            trainable=False,
            use_bias=False,
            weights=[hf],
        )(x_)
        d = Conv2D(
            1,
            [hs, vs],
            strides=1,
            padding="same",
            trainable=False,
            use_bias=False,
            weights=[df],
        )(x_)
        dr = Conv2D(
            1,
            [hs, vs],
            strides=1,
            padding="same",
            trainable=False,
            use_bias=False,
            weights=[dfr],
        )(x_)
        c = Add()([v, h, d, dr])
        c = Multiply()([xf, c])
        c = Conv2D(rbdims[i], 1, strides=1, padding="same", dilation_rate=1)(c)
        c = Concatenate(axis=3)([x_, c])
        x = Conv2D(rbdims[i], 3, strides=1, padding="same", dilation_rate=1)(c)

        print("ver: ", v.shape, h.shape, d.shape, dr.shape, c.shape, x.shape)

        # if i==0: break
    x = Conv2D(9, 1, strides=1, padding="same", dilation_rate=1)(x)
    logits_r = tf.keras.backend.resize_images(x, 2, 2, "channels_last")
    print(logits_cw.shape, logits_r.shape)
    # print(x_)
    # print(features_room_boundary)
    # pdb.set_trace()

    dfpmodel = Model(inputs=inp, outputs=[logits_cw, logits_r])
    print(dfpmodel.inputs, dfpmodel.outputs)
