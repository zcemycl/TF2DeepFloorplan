from argparse import Namespace

from dfp.convert2tflite import converter, parse_args


class fakeConverter:
    def __init__(self):
        self.optimizations = []
        self.experimental_new_converter = False

    def convert(self):
        return None


class fakeFile:
    def write(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class fakeModel:
    pass


def test_parse_args():
    args = parse_args(["--quantize"])
    assert args.quantize is True


def test_converter(mocker):
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
    )
    converter(args)
