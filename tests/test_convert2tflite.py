from argparse import Namespace
from types import TracebackType
from typing import Optional, Type

from pytest_mock import MockFixture

from dfp.convert2tflite import converter, parse_args


class fakeConverter:
    def __init__(self):
        self.optimizations = []
        self.experimental_new_converter = False

    def convert(self):
        return None


class fakeFile:
    def write(self, *args: str, **kwargs: int):
        pass

    def __enter__(self, *args: str, **kwargs: int):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        pass


class fakeModel:
    pass


def test_parse_args():
    args = parse_args(["--quantize"])
    assert args.quantize is True


def test_converter(mocker: MockFixture):
    model = fakeModel()
    con = fakeConverter()
    f = fakeFile()
    mocker.patch(
        "dfp.convert2tflite.tf.keras.models.load_model", return_value=model
    )
    mocker.patch(
        "dfp.convert2tflite.tf.lite.TFLiteConverter.from_keras_model",
        return_value=con,
    )
    mocker.patch("dfp.convert2tflite.open", return_value=f)
    args = Namespace(
        quantize=True,
        tflitedir="model/store/model.tflite",
        modeldir="model/store",
        compress_mode="quantization",
        tfmodel="subclass",
        loadmethod="pb",
    )
    converter(args)
