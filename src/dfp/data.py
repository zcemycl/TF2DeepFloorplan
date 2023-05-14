from typing import Dict, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf


def convert_one_hot_to_image(
    one_hot: tf.Tensor, dtype: str = "float", act: str = None
) -> tf.Tensor:
    if act == "softmax":
        one_hot = tf.keras.activations.softmax(one_hot)
    [n, h, w, c] = one_hot.shape.as_list()
    im = tf.reshape(tf.keras.backend.argmax(one_hot, axis=-1), [n, h, w, 1])
    if dtype == "int":
        im = tf.cast(im, dtype=tf.uint8)
    else:
        im = tf.cast(im, dtype=tf.float32)
    return im


def _parse_function(example_proto: bytes) -> Dict[str, tf.Tensor]:
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "boundary": tf.io.FixedLenFeature([], tf.string),
        "room": tf.io.FixedLenFeature([], tf.string),
        "door": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, feature)


def decodeAllRaw(
    x: Dict[str, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image = tf.io.decode_raw(x["image"], tf.uint8)
    boundary = tf.io.decode_raw(x["boundary"], tf.uint8)
    room = tf.io.decode_raw(x["room"], tf.uint8)
    return image, boundary, room


def preprocess(
    img: tf.Tensor, bound: tf.Tensor, room: tf.Tensor, size: int = 512
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, size, size, 3]) / 255
    bound = tf.reshape(bound, [-1, size, size])
    room = tf.reshape(room, [-1, size, size])
    hot_b = tf.one_hot(bound, 3, axis=-1)
    hot_r = tf.one_hot(room, 9, axis=-1)
    return img, bound, room, hot_b, hot_r


def loadDataset(size: int = 512) -> tf.data.Dataset:
    raw_dataset = tf.data.TFRecordDataset("r3d.tfrecords")
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def plotData(data: Dict[str, tf.Tensor]):
    img, bound, room = decodeAllRaw(data)
    img, bound, room, hb, hr = preprocess(img, bound, room)
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(bound[0].numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(convert_one_hot_to_image(hb)[0].numpy())


def main(dataset: tf.data.Dataset):
    for ite in range(2):
        for data in list(dataset.shuffle(400).batch(1)):
            plotData(data)
            plt.show()
            break


if __name__ == "__main__":
    dataset = loadDataset()
    main(dataset)
