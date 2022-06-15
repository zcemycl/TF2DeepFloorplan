import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from dfp.train import image_grid, parse_args, plot_to_image


class TestTrainCase:
    def test_image_grid(self):
        inp123 = tf.random.normal([1, 32, 32, 3])
        inp123 = tf.clip_by_value(inp123, 0, 1)
        inp45 = inp123 / tf.reduce_sum(inp123, axis=-1, keepdims=True)
        f = image_grid(inp123, inp123, inp123, inp45, inp45)
        assert isinstance(f, matplotlib.figure.Figure)

    def test_plot_to_image(self):
        f = plt.figure()
        img = plot_to_image(f)
        assert img.numpy().ndim == 4

    def test_parse_args(self):
        args = parse_args(["--lr", "1e-3", "--epochs", "300"])
        assert args.lr == 1e-3
        assert args.epochs == 300
