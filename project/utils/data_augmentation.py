import h5py
import numpy as np
import tensorflow as tf
import os

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing import image

import project.download_content as content

from project.model.loss import BBOX_REF

import ray

import logging

from operator import is_not
from functools import partial

logging.getLogger().setLevel(logging.INFO)

DATAPATH = os.path.join(content.DATAPATH, "MODEL",
                        os.getenv('DATA', 'part_data_300_vgg.h5'))

modelpath = os.path.join(content.DATAPATH, "MODEL")

ALL_ANCHORS = BBOX_REF.references.values

def get_group_imgs():
    with h5py.File(DATAPATH, 'r') as f:
        groups = list(f.keys())
    return groups


def load_data(group):
    """return an array with images infos"""
    with h5py.File(DATAPATH, 'r') as f:
        images = f['0']['images'][:]
        X = np.array([f['0'][i[0]][:] for i in images])

    return images, X


def match_bbox(bboxes):
    y = np.zeros((ALL_ANCHORS.shape[0], bboxes.shape[1] + 1))
    y[:, 0] = 1

    find_anchors = []
    for *classes, cx, cy, w, h in bboxes:
        ground_truth = np.array([cx, cy, w, h])
        anchors = BBOX_REF.match(ground_truth)

        find_anchors.append(anchors)

        for i in anchors:
            y[i] = [0] + classes + [cx, cy, w, h]

    for anchors in find_anchors:
        if len(anchors) > 0:
            return y
    raise ValueError("There are no bounding boxes match")


def resize(img, bboxes, proportion=.7, delta_x=0, delta_y=0):
    lb = (1.0 - proportion) / 2.0
    hb = 1.0 - lb

    if delta_x > lb or delta_y > lb:
        raise RuntimeError("Delta X or Delta Y can't be greater than the half of (1 - proportion)")

    new_bboxes = find_bbox(bboxes, proportion, delta_x, delta_y)

    return (tf.image.crop_and_resize(np.expand_dims(img, axis=0),
                                     [[delta_x + lb,
                                       delta_y + lb,
                                       delta_x + hb,
                                       delta_y + hb]],
                                     [0],
                                     [300, 300])[0],
            new_bboxes)


def resize_params(bboxes, proportion=.7, delta_x=0.0, delta_y=0.0):
    cx_0, cy_0, w_0, h_0 = bboxes

    step = (1.0 - proportion) / 2.0
    cx = (cx_0 - delta_y - step) / proportion
    cy = (cy_0 - delta_x - step) / proportion
    w = w_0 / proportion
    h = h_0 / proportion

    return [cx, cy, w, h]


def append_if_valid(bboxes, cx, cy, w, h, *classes):
    if cx > (1 - w/2) or cx < w/2 or w > 1 or w < 0:
        return
    if cy > (1 - h/2) or cy < h/2 or h > 1 or h < 0:
        return

    bboxes.append([*classes, cx, cy, w, h])


def find_bbox(bboxes, proportion=.7, delta_x=0.0, delta_y=0.0):
    new_bboxes = []
    for *classes, cx, cy, w, h in bboxes:
        n_cx, n_cy, n_w, n_h = resize_params([cx, cy, w, h], proportion, delta_x, delta_y)
        append_if_valid(new_bboxes, n_cx, n_cy, n_w, n_h, *classes)

    for ref_bboxes in new_bboxes:
        if len(ref_bboxes) > 0:
            return np.array(new_bboxes)
    raise ValueError('No bounding boxes in new image')


def normalize(img):
    return (img - np.mean(img)) / (np.std(img) + 1e-15)


def relevant_classes(y):
    # 0 is the no-class, -4 to -1 is the position of bounding box,
    # the other numbers can be understood in class-exploration notebook
    return y[:, :, [0, 14, 53, 64, 71, 91, 105, 116, 185, 195, 199, 203, 216,
                    257, 259, 261, 263, 264, 266, 267, 268, 269, 270, 271, 301,
                    320, 321, 380, 392, 465, 467, 483, 501, 513, 535, 552, 566,
                    581, 586, 593, -4, -3, -2, -1]]


def pre_process(img, y):
    img = np.expand_dims(img, axis=0)
    y = np.expand_dims(y, axis=0)

    return normalize(img), relevant_classes(y)
    # return normalize(img), y[:, :, [0, 53, 301, 465, -4, -3, -2, -1]]


def original_image(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        return pre_process(img, y)
    except ValueError:
        return None


@ray.remote
def async_original_image(*args):
    if args is None:
        return None
    return original_image(*args)


def flip_horiz(img, bboxes_raw):
    try:
        img_flip_h = np.flip(img, 1)

        bboxes = deepcopy(bboxes_raw)
        bboxes[:, -4] = 1 - bboxes[:, -4]
        y = match_bbox(bboxes)

        return pre_process(img_flip_h, y)
    except ValueError:
        return None


@ray.remote
def async_flip_horiz(img, bboxes):
    return flip_horiz(img, bboxes)


def flip_vert(img, bboxes_raw):
    try:
        img_flip_v = np.flip(img, 0)

        bboxes = deepcopy(bboxes_raw)
        bboxes[:, -3] = 1 - bboxes[:, -3]
        y = match_bbox(bboxes)

        return pre_process(img_flip_v, y)
    except ValueError:
        return None


@ray.remote
def async_flip_vert(img, bboxes):
    return flip_vert(img, bboxes)


def flip_both(img, bboxes_raw):
    try:
        img_flip_h_v = np.flip(img, [0, 1])

        bboxes = deepcopy(bboxes_raw)
        bboxes[:, -4:-2] = 1 - bboxes[:, -4:-2]
        y = match_bbox(bboxes)

        return pre_process(img_flip_h_v, y)
    except ValueError:
        return None


@ray.remote
def async_flip_both(img, bboxes):
    return flip_both(img, bboxes)


def zoom(img, bboxes_raw, proportion=.7, delta_x=0, delta_y=0):
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, proportion, delta_x, delta_y)

        y = match_bbox(bboxes)

        return pre_process(img_z, y)
    except ValueError:
        return None

@ray.remote
def async_zoom(img, bboxes, proportion, delta_x, delta_y):
    try:
        return zoom(img, bboxes, proportion, delta_x, delta_y)
    except ValueError:
        return None


def saturation(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        img_s = tf.image.random_saturation(img, .4, 1.6)
        return pre_process(img_s, y)
    except ValueError:
        return None


def contrast(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        img_c = tf.image.random_contrast(img, .5, .6)
        return pre_process(img_c, y)
    except ValueError:
        return None


@ray.remote
def batch_data_augmentation(batch):
    if batch:
        return [data_augmentation(*item) for item in batch if item is not None]
    return []


def data_augmentation(image_info, bboxes):
    img_bin = image.load_img('project/' + image_info[1], target_size=(300, 300))
    img = image.img_to_array(img_bin)

    results = [original_image(img, bboxes),
               flip_horiz(img, bboxes),
               flip_vert(img, bboxes),
               flip_both(img, bboxes),
               saturation(img, bboxes),
               contrast(img, bboxes),
               zoom(img, bboxes, .8, -.099, -.099),
               zoom(img, bboxes, .8,     0, -.099),
               zoom(img, bboxes, .8,  .099, -.099),
               zoom(img, bboxes, .8, -.099, 0),
               zoom(img, bboxes, .8,     0, 0),
               zoom(img, bboxes, .8,  .099, 0),
               zoom(img, bboxes, .8, -.099, .099),
               zoom(img, bboxes, .8,     0, .099),
               zoom(img, bboxes, .8,  .099, .099),
               zoom(img, bboxes, .6, -.2, -.2),
               zoom(img, bboxes, .6,   0, -.2),
               zoom(img, bboxes, .6,  .2, -.2),
               zoom(img, bboxes, .6, -.2, 0),
               zoom(img, bboxes, .6,   0, 0),
               zoom(img, bboxes, .6,  .2, 0),
               zoom(img, bboxes, .6, -.2, .2),
               zoom(img, bboxes, .6,   0, .2),
               zoom(img, bboxes, .6,  .2, .2)]

    return list(filter(partial(is_not, None), results))


def main():
    def gen_data():
        for group in get_group_imgs():
            images, X = load_data(group)

            for image_info, bboxes in zip(images, X):
                data_generator = data_augmentation(image_info, bboxes)

                try:
                    img, y = next(data_generator)

                    img = np.expand_dims(img, axis=0)
                    y = np.expand_dims(y, axis=0)

                    # normalize pixels
                    yield ((img - np.mean(img)) / (np.std(img) + 1e-15)), y
                except StopIteration:
                    break

    # setting path to data
    datapath = os.path.join(modelpath, "data_augmentation_300_vgg.h5")

    # open file and maintain it opened
    f = h5py.File(datapath, 'w')

    # size of the hfd5 group
    group_size = 10000

    try:
        g = gen_data()

        data_ref = []

        try:
            for i, (x, y) in tqdm(enumerate(g)):
                if i % group_size == 0:
                    group = str(int(i/group_size))
                    f.create_group(group)

                f[group].create_dataset(name=f"{i}",
                                        data=x[0],
                                        dtype=np.float16,
                                        compression='gzip',
                                        compression_opts=9)

                f[group].create_dataset(name=f"{i}-loc",
                                        data=y[0][:, -4:],
                                        dtype=np.float16,
                                        compression='gzip',
                                        compression_opts=9)

                f[group].create_dataset(name=f"{i}-conf",
                                        data=y[0][:, :-4],
                                        dtype=np.int8,
                                        compression='gzip',
                                        compression_opts=9)

                data_ref.append([f"{i}-loc", f"{i}-conf", f"{i}"])

                if (i+1) % group_size == 0:
                    f[group].create_dataset(name='data',
                                            data=data_ref,
                                            dtype=h5py.special_dtype(vlen=str))
                    data_ref = []

        except StopIteration:
            f[group].create_dataset(name='data',
                                    data=data_ref,
                                    dtype=h5py.special_dtype(vlen=str))

    finally:
        f.close()


if __name__ == '__main__':
    main()
