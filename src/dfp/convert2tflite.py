import argparse
import sys
import tempfile
from typing import List

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

from .data import decodeAllRaw, loadDataset, preprocess
from .loss import balanced_entropy, cross_two_tasks_weight
from .net_func import deepfloorplanFunc


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
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--modeldir", type=str, default="model/store")
    p.add_argument("--tflitedir", type=str, default="model/store/model.tflite")
    p.add_argument("--quantize", action="store_true")
    p.add_argument(
        "--compress-mode",
        type=str,
        default="quantization",
        choices=["quantization", "prune", "cluster"],
    )
    p.add_argument(
        "--loadmethod",
        type=str,
        default="log",
        choices=["log", "tflite", "pb", "none"],
    )  # log,tflite,pb
    return p.parse_args(args)


def prune(config: argparse.Namespace):
    if config.loadmethod == "log":
        base_model = deepfloorplanFunc()
        base_model.load_weights(config.modeldir)
    elif config.loadmethod == "pb":
        base_model = tf.keras.models.load_model(config.modeldir)
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)
    dataset = loadDataset()
    optimizer = tf.keras.optimizers.Adam()
    log_dir = tempfile.mkdtemp()
    print(f"log directory: {log_dir}...")
    unused_arg = -1
    epochs = 2
    batches = 1

    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(
        log_dir=log_dir
    )  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)

    step_callback.on_train_begin()  # run pruning callback
    for _ in range(epochs):
        log_callback.on_epoch_begin(epoch=unused_arg)  # run pruning callback
        for data in tqdm(list(dataset.shuffle(400).batch(batches))):
            step_callback.on_train_batch_begin(
                batch=unused_arg
            )  # run pruning callback
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)

            with tf.GradientTape() as tape:
                logits_r, logits_cw = model_for_pruning(img, training=True)
                loss1 = balanced_entropy(logits_r, hr)
                loss2 = balanced_entropy(logits_cw, hb)
                w1, w2 = cross_two_tasks_weight(hr, hb)
                loss_value = w1 * loss1 + w2 * loss2
                grads = tape.gradient(
                    loss_value, model_for_pruning.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(grads, model_for_pruning.trainable_variables)
                )

        step_callback.on_epoch_end(batch=unused_arg)  # run pruning callback


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.compress_mode == "quantization" or args.quantize:
        converter(args)
    if args.tfmodel == "func":
        if args.compress_mode == "prune":
            prune(args)
        elif args.compress_mode == "cluster":
            pass
    elif args.tfmodel == "subclass":
        raise Exception(
            "Pruning or Clustering for Subclass Model are not available."
        )
