import argparse
import io
import os
import sys
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from .data import (
    convert_one_hot_to_image,
    decodeAllRaw,
    loadDataset,
    preprocess,
)
from .loss import balanced_entropy, cross_two_tasks_weight
from .net import deepfloorplanModel
from .net_func import deepfloorplanFunc
from .utils.settings import overwrite_args_with_toml

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def init(
    config: argparse.Namespace,
) -> Tuple[tf.data.Dataset, tf.keras.Model, tf.keras.optimizers.Optimizer]:
    dataset = loadDataset()
    if config.tfmodel == "subclass":
        model = deepfloorplanModel(config=config)
    elif config.tfmodel == "func":
        model = deepfloorplanFunc(config=config)
    os.system(f"mkdir -p {config.modeldir}")
    if config.weight:
        model.load_weights(config.weight)
    # optim = tf.keras.optimizers.AdamW(learning_rate=config.lr,
    #   weight_decay=config.wd)
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)
    return dataset, model, optim


def plot_to_image(figure: matplotlib.figure.Figure) -> tf.Tensor:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(
    img: tf.Tensor,
    bound: tf.Tensor,
    room: tf.Tensor,
    logr: tf.Tensor,
    logcw: tf.Tensor,
) -> matplotlib.figure.Figure:
    figure = plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img[0].numpy())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 2)
    plt.imshow(bound[0].numpy())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 3)
    plt.imshow(room[0].numpy())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 5)
    plt.imshow(convert_one_hot_to_image(logcw)[0].numpy().squeeze())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 6)
    plt.imshow(convert_one_hot_to_image(logr)[0].numpy().squeeze())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    return figure


@tf.function
def train_step(
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    img: tf.Tensor,
    hr: tf.Tensor,
    hb: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # forward
    with tf.GradientTape() as tape:
        logits_r, logits_cw = model(img)
        loss1 = balanced_entropy(logits_r, hr)
        loss2 = balanced_entropy(logits_cw, hb)
        w1, w2 = cross_two_tasks_weight(hr, hb)
        loss = w1 * loss1 + w2 * loss2
    # backward
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return logits_r, logits_cw, loss, loss1, loss2


def main(config: argparse.Namespace):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, optim = init(config)
    # training loop
    for epoch in range(config.epochs):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.shuffle(400).batch(config.batchsize))):
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            logits_r, logits_cw, loss, loss1, loss2 = train_step(
                model, optim, img, hr, hb
            )

            # plot progress
            if pltiter % config.saveTensorInterval == 0:
                f = image_grid(img, bound, room, logits_r, logits_cw)
                im = plot_to_image(f)
                with writer.as_default():
                    tf.summary.scalar("Loss", loss.numpy(), step=pltiter)
                    tf.summary.scalar("LossR", loss1.numpy(), step=pltiter)
                    tf.summary.scalar("LossB", loss2.numpy(), step=pltiter)
                    tf.summary.image("Data", im, step=pltiter)
                writer.flush()
            pltiter += 1

        # save model
        if epoch % config.saveModelInterval == 0:
            model.save_weights(config.logdir + "/G")
            model.save(config.modeldir)
            print("[INFO] Saving Model ...")


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--batchsize", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--logdir", type=str, default="log/store")
    p.add_argument("--modeldir", type=str, default="model/store")
    p.add_argument("--weight", type=str)
    p.add_argument("--saveTensorInterval", type=int, default=10)
    p.add_argument("--saveModelInterval", type=int, default=20)
    p.add_argument("--tomlfile", type=str, default=None)
    p.add_argument(
        "--feature-channels",
        type=int,
        action="store",
        default=[256, 128, 64, 32],
        nargs=4,
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=["vgg16", "resnet50", "mobilenetv1", "mobilenetv2"],
    )
    p.add_argument(
        "--feature-names",
        type=str,
        action="store",
        nargs=5,
        default=[
            "block1_pool",
            "block2_pool",
            "block3_pool",
            "block4_pool",
            "block5_pool",
        ],
    )
    # mobilenetv2
    # layer_names = [
    #     "block_1_expand_relu",  # 256x256x96
    #     "block_3_expand_relu",  # 128x128x144
    #     "block_5_expand_relu",  # 64x64x192
    #     "block_13_expand_relu",  # 32x32x576
    #     "out_relu",  # 16x16x1280
    # ]
    # mobilenet
    # layer_names = [
    #     "conv_pw_1_relu",  # 256x256x64
    #     "conv_pw_3_relu",  # 128x128x128
    #     "conv_pw_5_relu",  # 64x64x256
    #     "conv_pw_7_relu",  # 32x32x512
    #     # "conv_pw_13_relu",  # 16x16x1024
    #     "conv_pw_12_relu",  # 16x16x1024
    # ]
    # resnet50
    # layer_names = [
    #     "conv1_relu",  # 256x256x64
    #     "conv2_block3_out",  # 128x128x256
    #     "conv3_block4_out",  # 64x64x512
    #     "conv4_block6_out",  # 32x32x1024
    #     "conv5_block3_out",  # 16x16x2048
    # ]
    return p.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    print(args)
    main(args)
