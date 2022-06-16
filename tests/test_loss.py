import unittest

import tensorflow as tf

from dfp.loss import balanced_entropy, cross_two_tasks_weight


class TestLossCase(unittest.TestCase):
    randx = tf.random.normal((1, 32, 32, 3), 0, 1)
    randy = tf.random.normal((1, 32, 32, 9), 0, 1)

    def test_balanced_entropy(self):
        loss = balanced_entropy(self.__class__.randx, self.__class__.randx)
        self.assertEqual(loss.dtype, tf.float32)

    def test_weight(self):
        w1, w2 = cross_two_tasks_weight(
            self.__class__.randx, self.__class__.randy
        )
        w = w1 + w2
        self.assertLessEqual(w.numpy(), 2.0)


if __name__ == "__main__":
    unittest.main()
