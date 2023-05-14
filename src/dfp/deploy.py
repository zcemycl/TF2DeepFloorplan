import argparse
import gc
import os
import sys

# import pdb
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .data import convert_one_hot_to_image
from .net import deepfloorplanModel
from .utils.rgb_ind_convertor import (
    floorplan_boundary_map,
    floorplan_fuse_map,
    ind2rgb,
)
from .utils.util import fill_break_line, flood_fill, refine_room_region

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def init(
    config: argparse.Namespace,
) -> Tuple[tf.keras.Model, tf.Tensor, np.ndarray]:
    model = deepfloorplanModel()
    if config.loadmethod == "log":
        model.load_weights(config.weight)
    elif config.loadmethod == "pb":
        model = tf.keras.models.load_model(config.weight)
    elif config.loadmethod == "tflite":
        model = tf.lite.Interpreter(model_path=config.weight)
        model.allocate_tensors()
    img = mpimg.imread(config.image)
    shp = img.shape
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, 512, 512, 3]) / 255
    if config.loadmethod == "tflite":
        return model, img, shp
    model.trainable = False
    model.vgg16.trainable = False
    return model, img, shp


def predict(
    model: tf.keras.Model, img: tf.Tensor, shp: np.ndarray
) -> Tuple[tf.Tensor, tf.Tensor]:
    features = []
    feature = img
    for layer in model.vgg16.layers:
        feature = layer(feature)
        if layer.name.find("pool") != -1:
            features.append(feature)
    x = feature
    features = features[::-1]
    del model.vgg16
    gc.collect()

    featuresrbp = []
    for i in range(len(model.rbpups)):
        x = model.rbpups[i](x) + model.rbpcv1[i](features[i + 1])
        x = model.rbpcv2[i](x)
        featuresrbp.append(x)
    logits_cw = tf.keras.backend.resize_images(
        model.rbpfinal(x), 2, 2, "channels_last"
    )

    x = features.pop(0)
    nLays = len(model.rtpups)
    for i in range(nLays):
        rs = model.rtpups.pop(0)
        r1 = model.rtpcv1.pop(0)
        r2 = model.rtpcv2.pop(0)
        f = features.pop(0)
        x = rs(x) + r1(f)
        x = r2(x)
        a = featuresrbp.pop(0)
        x = model.non_local_context(a, x, i)

    del featuresrbp
    logits_r = tf.keras.backend.resize_images(
        model.rtpfinal(x), 2, 2, "channels_last"
    )
    del model.rtpfinal

    return logits_cw, logits_r


def post_process(
    rm_ind: np.ndarray, bd_ind: np.ndarray, shp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)
    cw_mask = np.reshape(cw_mask, (*shp[:2], -1))
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask.reshape(*shp[:2], -1) * new_rm_ind
    new_bd_ind = fill_break_line(bd_ind).squeeze()
    return new_rm_ind, new_bd_ind


def colorize(r: np.ndarray, cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cr = ind2rgb(r, color_map=floorplan_fuse_map)
    ccw = ind2rgb(cw, color_map=floorplan_boundary_map)
    return cr, ccw


def main(config: argparse.Namespace) -> np.ndarray:
    model, img, shp = init(config)
    if config.loadmethod == "log":
        logits_cw, logits_r = predict(model, img, shp)
    elif config.loadmethod == "pb" or config.loadmethod == "none":
        logits_r, logits_cw = model(img)
    elif config.loadmethod == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        logits_r = model.get_tensor(output_details[0]["index"])
        logits_cw = model.get_tensor(output_details[1]["index"])
        logits_cw = tf.convert_to_tensor(logits_cw)
        logits_r = tf.convert_to_tensor(logits_r)
    logits_r = tf.image.resize(logits_r, shp[:2])
    logits_cw = tf.image.resize(logits_cw, shp[:2])
    r = convert_one_hot_to_image(logits_r)[0].numpy()
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()

    if not config.colorize and not config.postprocess:
        cw[cw == 1] = 9
        cw[cw == 2] = 10
        r[cw != 0] = 0
        return (r + cw).squeeze()
    elif config.colorize and not config.postprocess:
        r_color, cw_color = colorize(r.squeeze(), cw.squeeze())
        return r_color + cw_color

    newr, newcw = post_process(r, cw, shp)
    if not config.colorize and config.postprocess:
        newcw[newcw == 1] = 9
        newcw[newcw == 2] = 10
        newr[newcw != 0] = 0
        return newr.squeeze() + newcw
    newr_color, newcw_color = colorize(newr.squeeze(), newcw.squeeze())
    result = newr_color + newcw_color
    print(shp, result.shape)

    if config.save:
        mpimg.imsave(config.save, result.astype(np.uint8))

    return result


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default="resources/30939153.jpg")
    p.add_argument("--weight", type=str, default="log/store/G")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--colorize", action="store_true")
    p.add_argument(
        "--loadmethod",
        type=str,
        default="log",
        choices=["log", "tflite", "pb", "none"],
    )  # log,tflite,pb
    p.add_argument("--save", type=str)
    return p.parse_args(args)


def deploy_plot_res(result: np.ndarray):
    print(result.shape)
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    result = main(args)
    deploy_plot_res(result)
    plt.show()
    # pdb.set_trace()
