import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage


def fast_hist(im: np.ndarray, gt: np.ndarray, n: int = 9) -> np.ndarray:
    """
    n is num_of_classes
    """
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + im[k], minlength=n**2
    ).reshape(n, n)


def flood_fill(test_array: np.ndarray, h_max: int = 255) -> np.ndarray:
    """
    fill in the hole
    """
    test_array = test_array.squeeze()
    input_array = np.copy(test_array)
    el = ndimage.generate_binary_structure(2, 2).astype(int)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = ndimage.generate_binary_structure(2, 1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(
            input_array,
            ndimage.grey_erosion(output_array, size=(3, 3), footprint=el),
        )
    return output_array


def fill_break_line(cw_mask: np.ndarray) -> np.ndarray:
    broken_line_h = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    broken_line_h2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    broken_line_v = np.transpose(broken_line_h)
    broken_line_v2 = np.transpose(broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v2)

    return cw_mask


def refine_room_region(cw_mask: np.ndarray, rm_ind: np.ndarray) -> np.ndarray:
    label_rm, num_label = ndimage.label((1 - cw_mask))
    new_rm_ind = np.zeros(rm_ind.shape)
    for j in range(1, num_label + 1):
        mask = (label_rm == j).astype(np.uint8)
        ys, xs, _ = np.where(mask != 0)
        area = (np.amax(xs) - np.amin(xs)) * (np.amax(ys) - np.amin(ys))
        if area < 100:
            continue
        else:
            room_types, type_counts = np.unique(
                mask * rm_ind, return_counts=True
            )
            if len(room_types) > 1:
                room_types = room_types[1:]
                # ignore background type which is zero
                type_counts = type_counts[1:]
                # ignore background count
            new_rm_ind += mask * room_types[np.argmax(type_counts)]

    return new_rm_ind


def print_model_weights_sparsity(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            if "kernel" not in weight.name or "centroid" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            print(
                f"{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                f"({zero_num}/{weight_size})",
            )


def print_model_weight_clusters(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            # ignore auxiliary quantization weights
            if "quantize_layer" in weight.name:
                continue
            if "kernel" in weight.name:
                unique_count = len(np.unique(weight))
                print(f"{layer.name}/{weight.name}: {unique_count} clusters ")
