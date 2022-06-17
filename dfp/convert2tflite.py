import argparse
import sys
from typing import List

import tensorflow as tf


def converter(config: argparse.Namespace):
    model = tf.keras.models.load_model(config.modeldir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if config.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    with open(config.tflitedir, "wb") as f:
        f.write(tflite_model)


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modeldir", type=str, default="model/store")
    p.add_argument("--tflitedir", type=str, default="model/store/model.tflite")
    p.add_argument("--quantize", action="store_true")
    return p.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    converter(args)
