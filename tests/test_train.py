from argparse import Namespace
from types import TracebackType
from typing import Any, List, Optional, Tuple, Type

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from pytest_mock import MockFixture

from dfp.train import image_grid, init, parse_args, plot_to_image, train_step


class fakeModel:
    def __init__(self):
        self.trainable_weights = []

    def load_weights(self, weights: str):
        pass

    def __call__(
        self, *args: str, **kwargs: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        a = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
        b = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
        return a / tf.reduce_sum(a, axis=-1, keepdims=True), b / tf.reduce_sum(
            b, axis=-1, keepdims=True
        )


class fakeOptim:
    def apply_gradients(self, *args: str, **kwargs: int):
        pass


class fakeTape:
    def GradientTape(self) -> object:
        return self

    def gradient(self, *args: str, **kwargs: int) -> List[Any]:
        return []

    def __enter__(self) -> object:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        pass


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

    def test_init(self, mocker: MockFixture):
        model = fakeModel()
        mocker.patch("dfp.train.loadDataset", return_value=None)
        mocker.patch("dfp.train.deepfloorplanModel", return_value=model)
        args = Namespace(weight="fakepath", lr=1e-4)
        ds, model, opt = init(args)
        assert isinstance(opt, tf.keras.optimizers.Optimizer)

    def test_trainstep(self, mocker: MockFixture):
        model = fakeModel()
        tape = fakeTape()
        optim = fakeOptim()
        img = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
        hr = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
        hb = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
        mocker.patch("dfp.train.tf.GradientTape", return_value=tape)
        logits_r, logits_cw, loss, loss1, loss2 = train_step.__wrapped__(
            model, optim, img, hr, hb
        )
        assert logits_r.numpy().shape == (1, 16, 16, 3)
        assert logits_cw.numpy().shape == (1, 16, 16, 9)
