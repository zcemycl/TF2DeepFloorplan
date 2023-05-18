from argparse import Namespace
from typing import Dict, List, Tuple

import pytest

import numpy as np
import tensorflow as tf
from pytest_mock import MockFixture

from dfp.deploy import (
    colorize,
    deploy_plot_res,
    init,
    main,
    parse_args,
    post_process,
    predict,
)


class fakeLayer:
    def __init__(self):
        self.name = "pool"

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return x


class fakeVGG16:
    def __init__(self):
        self.trainable = True
        self.layer = fakeLayer()
        self.layers = [self.layer, self.layer]


class fakeModel:
    def __init__(self):
        self.trainable = True
        self.vgg16 = fakeVGG16()
        self.rtpfinal = fakeLayer()
        self.rbpups = [self.rbpfinal]
        self.rbpcv1 = [self.rbpfinal]
        self.rbpcv2 = [self.rbpfinal]
        self.rtpups = [self.rtpfinal]
        self.rtpcv1 = [self.rtpfinal]
        self.rtpcv2 = [self.rtpfinal]

    def rbpfinal(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def non_local_context(
        self, x: tf.Tensor, *args: int, **kwargs: str
    ) -> tf.Tensor:
        return x

    def predict(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        a = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
        b = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
        return b / tf.reduce_sum(b, axis=-1, keepdims=True), a / tf.reduce_sum(
            a, axis=-1, keepdims=True
        )

    def invoke(self):
        pass

    def get_input_details(self) -> List[Dict[str, Tuple[int, int, int]]]:
        return [{"index": (512, 512, 3)}]

    def get_output_details(self) -> List[Dict[str, Tuple[int, int, int]]]:
        return [{"index": (512, 512, 9)}, {"index": (512, 512, 3)}]

    def set_tensor(self, ind: Tuple[int, int, int], img: tf.Tensor):
        pass

    def get_tensor(self, ind: Tuple[int, int, int]):
        if ind[2] == 3:
            a = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
            return a / tf.reduce_sum(a, axis=-1, keepdims=True)
        elif ind[2] == 9:
            b = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
            return b / tf.reduce_sum(b, axis=-1, keepdims=True)

    def load_weights(self, x: str):
        pass

    def allocate_tensors(self):
        pass


@pytest.fixture
def model_img() -> Tuple[fakeModel, tf.Tensor]:
    model = fakeModel()
    img = tf.random.normal((1, 16, 16, 3))
    return model, img


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_colorize(h: int, w: int, c: int, mocker: MockFixture):
    inp = np.random.randint(2, size=(h, w))
    out = np.zeros((h, w, c))
    m = mocker.patch("dfp.deploy.ind2rgb")
    m.return_value = out
    r, cw = colorize(inp, inp)
    assert r.shape == (h, w, c) and cw.shape == (h, w, c)


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_post_process(h: int, w: int, c: int):
    inp = np.ones((h, w, 1))
    r, cw = post_process(inp, inp, np.array([h, w, c]))
    assert r.shape == (h, w, 1) and cw.shape == (h, w)


def test_parse_args():
    args = parse_args(
        ["--postprocess", "--loadmethod", "tflite", "--tfmodel", "subclass"]
    )
    assert args.postprocess is True
    assert args.loadmethod == "tflite"


def test_init_none(mocker: MockFixture):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    args = parse_args(
        '--loadmethod none --image "" --tfmodel subclass'.split()
    )
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_log(mocker: MockFixture):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    args = parse_args(
        """--loadmethod log --image ""
--weight log/store/G --tfmodel subclass""".split()
    )
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_pb(mocker: MockFixture):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    mocker.patch("dfp.deploy.tf.keras.models.load_model", return_value=model)
    args = parse_args(
        """--loadmethod pb --image ""
--weight model/store --tfmodel subclass""".split()
    )
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_tflite(mocker: MockFixture):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    mocker.patch("dfp.deploy.tf.lite.Interpreter", return_value=model)
    args = parse_args(
        """--loadmethod tflite --image \"\"
--weight model/store/model.tflite
--tfmodel subclass""".split()
    )
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


@pytest.mark.parametrize(
    "colorize,postprocess,expected",
    [
        (True, True, (16, 16, 3)),
        (False, False, (16, 16)),
        (True, False, (16, 16, 3)),
        (False, True, (16, 16)),
    ],
)
def test_main(
    colorize: bool,
    postprocess: bool,
    expected: Tuple[int, int, int],
    model_img: Tuple[fakeModel, tf.Tensor],
    mocker: MockFixture,
):
    args = Namespace(
        loadmethod="none",
        image="",
        colorize=colorize,
        postprocess=postprocess,
        save=True,
        tfmodel="subclass",
    )
    model, img = model_img

    mocker.patch(
        "dfp.deploy.init", return_value=(model.predict, img, [16, 16, 3])
    )
    mocker.patch("dfp.deploy.mpimg.imsave", return_value=None)
    res = main(args)
    assert res.shape == expected


def test_main_tflite(
    model_img: Tuple[fakeModel, tf.Tensor], mocker: MockFixture
):
    args = Namespace(
        loadmethod="tflite",
        image="",
        colorize=True,
        postprocess=True,
        save=False,
        tfmodel="subclass",
    )
    model, img = model_img
    mocker.patch("dfp.deploy.init", return_value=(model, img, [16, 16, 3]))
    res = main(args)
    assert res.shape == (16, 16, 3)


def test_main_log(model_img: Tuple[fakeModel, tf.Tensor], mocker: MockFixture):
    args = Namespace(
        loadmethod="log",
        image="",
        colorize=True,
        postprocess=True,
        save=False,
        tfmodel="subclass",
    )
    model, img = model_img
    mocker.patch("dfp.deploy.init", return_value=(model, img, [16, 16, 3]))
    mocker.patch(
        "dfp.deploy.predict",
        return_value=(
            tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1),
            tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1),
        ),
    )
    res = main(args)
    assert res.shape == (16, 16, 3)


def test_deploy_plot_res():
    a = np.zeros((16, 16, 3))
    deploy_plot_res(a)


def test_predict(model_img: Tuple[fakeModel, tf.Tensor], mocker: MockFixture):
    model, img = model_img
    shp = np.array([16, 16, 3])
    a, b = predict(model, img, shp)
    assert a.numpy().shape == (1, 32, 32, 3)
