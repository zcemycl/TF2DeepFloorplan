from argparse import Namespace

import pytest

import numpy as np
import tensorflow as tf

from dfp.deploy import (
    colorize,
    deploy_plot_res,
    init,
    main,
    parse_args,
    post_process,
)


class fakeVGG16:
    def __init__(self):
        self.trainable = True


class fakeModel:
    def __init__(self):
        self.trainable = True
        self.vgg16 = fakeVGG16()

    def predict(self, x):
        a = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
        b = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
        return b / tf.reduce_sum(b, axis=-1, keepdims=True), a / tf.reduce_sum(
            a, axis=-1, keepdims=True
        )

    def invoke(self):
        pass

    def get_input_details(self):
        return [{"index": (512, 512, 3)}]

    def get_output_details(self):
        return [{"index": (512, 512, 9)}, {"index": (512, 512, 3)}]

    def set_tensor(self, ind, img):
        pass

    def get_tensor(self, ind):
        if ind[2] == 3:
            a = tf.random.uniform((1, 16, 16, 3), minval=0, maxval=1)
            return a / tf.reduce_sum(a, axis=-1, keepdims=True)
        elif ind[2] == 9:
            b = tf.random.uniform((1, 16, 16, 9), minval=0, maxval=1)
            return b / tf.reduce_sum(b, axis=-1, keepdims=True)

    def load_weights(self, x):
        pass

    def allocate_tensors(self):
        pass


@pytest.fixture
def model_img():
    model = fakeModel()
    img = tf.random.normal((1, 16, 16, 3))
    return model, img


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_colorize(h, w, c, mocker):
    inp = np.random.randint(2, size=(h, w))
    out = np.zeros((h, w, c))
    m = mocker.patch("dfp.deploy.ind2rgb")
    m.return_value = out
    r, cw = colorize(inp, inp)
    assert r.shape == (h, w, c) and cw.shape == (h, w, c)


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_post_process(h, w, c):
    inp = np.ones((h, w, 1))
    r, cw = post_process(inp, inp, [h, w, c])
    assert r.shape == (h, w, 1) and cw.shape == (h, w)


def test_parse_args():
    args = parse_args(["--postprocess", "--loadmethod", "tflite"])
    assert args.postprocess is True
    assert args.loadmethod == "tflite"


def test_init_none(mocker):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    args = Namespace(loadmethod="none", image="")
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_log(mocker):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    args = Namespace(loadmethod="log", image="", weight="log/store/G")
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_pb(mocker):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    mocker.patch("dfp.deploy.tf.keras.models.load_model", return_value=model)
    args = Namespace(loadmethod="pb", image="", weight="model/store")
    model_, img, shp = init(args)
    assert shp == (16, 16, 3)


def test_init_tflite(mocker):
    model = fakeModel()
    mocker.patch("dfp.deploy.deepfloorplanModel", return_value=model)
    mocker.patch("dfp.deploy.mpimg.imread", return_value=np.zeros([16, 16, 3]))
    mocker.patch("dfp.deploy.tf.lite.Interpreter", return_value=model)
    args = Namespace(
        loadmethod="tflite", image="", weight="model/store/model.tflite"
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
def test_main(colorize, postprocess, expected, model_img, mocker):
    args = Namespace(
        loadmethod="none",
        image="",
        colorize=colorize,
        postprocess=postprocess,
        save=True,
    )
    model, img = model_img

    mocker.patch(
        "dfp.deploy.init", return_value=(model.predict, img, [16, 16, 3])
    )
    mocker.patch("dfp.deploy.mpimg.imsave", return_value=None)
    res = main(args)
    assert res.shape == expected


def test_main_tflite(model_img, mocker):
    args = Namespace(
        loadmethod="tflite",
        image="",
        colorize=True,
        postprocess=True,
        save=False,
    )
    model, img = model_img
    mocker.patch("dfp.deploy.init", return_value=(model, img, [16, 16, 3]))
    res = main(args)
    assert res.shape == (16, 16, 3)


def test_main_log(model_img, mocker):
    args = Namespace(
        loadmethod="log",
        image="",
        colorize=True,
        postprocess=True,
        save=False,
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
