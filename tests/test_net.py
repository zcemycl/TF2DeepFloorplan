import unittest

import numpy as np
import tensorflow as tf
from dfp.net import deepfloorplanModel
from tensorflow.keras.applications.vgg16 import preprocess_input


class TestCase(unittest.TestCase):
    def test_shape(self):
        x = np.random.randn(1, 512, 512, 3)
        with tf.device("/cpu:0"):
            x = preprocess_input(x)
            model = deepfloorplanModel()
            logits_r, logits_cw = model(x)
        self.assertEqual(logits_r.numpy().shape, (1, 512, 512, 9))
        self.assertEqual(logits_cw.numpy().shape, (1, 512, 512, 3))


if __name__ == "__main__":
    unittest.main()
