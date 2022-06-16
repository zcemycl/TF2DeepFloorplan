from dfp.convert2tflite import parse_args


def test_parse_args():
    args = parse_args(["--quantize"])
    assert args.quantize is True
