import unittest

from net import *


class TestCase(unittest.TestCase):
    def test_shape(self):
        x = np.random.randn(1,512,512,3)
        with tf.device('/cpu:0'):
            x = preprocess_input(x)
            model = deepfloorplanModel()
            logits_r,logits_cw = model(x)
        self.assertEqual(logits_r.numpy().shape,(1,512,512,9))
        self.assertEqual(logits_cw.numpy().shape,(1,512,512,3))

if __name__ == "__main__":
    unittest.main()
