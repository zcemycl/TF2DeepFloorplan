import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from dfp.net import (
    conv2d,
    deepfloorplanModel,
    max_pool2d,
    up_bilinear,
    upconv2d,
)


class TestNetCase(unittest.TestCase):
    model = deepfloorplanModel()
    randx = np.random.randn(1, 512, 512, 3)

    def test_deepfloorplan_forward(self):
        with tf.device("/cpu:0"):
            x = preprocess_input(self.__class__.randx)
            logits_r, logits_cw = self.__class__.model(x)
            shpr, shpcw = logits_r.numpy().shape, logits_cw.numpy().shape
        self.assertEqual((shpr, shpcw), ((1, 512, 512, 9), (1, 512, 512, 3)))

    def test_vgg16(self):
        gt = [
            (1, 256, 256, 64),
            (1, 128, 128, 128),
            (1, 64, 64, 256),
            (1, 32, 32, 512),
            (1, 16, 16, 512),
        ]
        vgg16 = self.__class__.model.vgg16
        out = []
        x = self.__class__.randx
        for lay in vgg16.layers:
            x = lay(x)
            if lay.name.find("pool") != -1:
                out.append(x.shape)
        self.assertEqual(out, gt)

    def test_conv2d(self):
        lay = conv2d(1, act="leaky")
        x = np.random.randn(1, 32, 32, 2)
        x = lay(x)
        self.assertEqual(x.numpy().shape, (1, 32, 32, 1))

    def test_upconv2d(self):
        lay = upconv2d(1, act="relu")
        x = np.random.randn(1, 16, 16, 2)
        x = lay(x)
        self.assertEqual(x.numpy().shape, (1, 32, 32, 1))

    def test_maxpool2d(self):
        lay = max_pool2d()
        x = np.random.randn(1, 32, 32, 1)
        x = lay(x)
        self.assertEqual(x.numpy().shape, (1, 16, 16, 1))

    def test_upbilinear(self):
        lay = up_bilinear(2)
        x = np.random.randn(1, 16, 16, 1)
        x = lay(x)
        self.assertEqual(x.numpy().shape, (1, 16, 16, 2))


if __name__ == "__main__":
    unittest.main()
