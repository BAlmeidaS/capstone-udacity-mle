import h5py
import numpy as np
import tensorflow as tf
import os

from copy import deepcopy

from keras.preprocessing import image

import project.download_content as content

from project.model.loss import BBOX_REF

import logging

from operator import is_not
from functools import partial

from project.utils.data_bbox_match_hdf5 import load_ohc

import json


logging.getLogger().setLevel(logging.INFO)

DATAPATH = os.path.join(content.DATAPATH, "MODEL",
                        os.getenv('DATA', 'images_bboxes.h5'))

modelpath = os.path.join(content.DATAPATH, "MODEL")

ALL_ANCHORS = BBOX_REF.references.values

OHC = load_ohc()

with open(os.path.join(content.DATAPATH, 'data_aug.json')) as f:
    AUG_CRITERIA = json.load(f)


def get_group_imgs():
    with h5py.File(DATAPATH, 'r') as f:
        groups = list(f.keys())
    return groups


def load_data(group):
    """return an array with images infos"""
    with h5py.File(DATAPATH, 'r') as f:
        images = f[group]['images'][:]
        X = np.array([f[group][i[0]][:] for i in images])

    return images, X


def match_bbox(bboxes, unmatch_return=False):
    # -4 to -1 is the position of bounding box,
    # the other numbers can be understood in class-exploration notebook
    y = np.zeros((ALL_ANCHORS.shape[0], bboxes.shape[1] + 1))
    y[:, 0] = 1

    iou_threshold = np.random.choice([.7]*1 + [.6]*3 + [.5]*8 + [.4]*3 + [.3]*1)

    find_anchors = []

    for *classes, cx, cy, w, h in bboxes:
        if sum(classes) > 0:
            ground_truth = np.array([cx, cy, w, h])
            anchors = BBOX_REF.match(ground_truth, iou_threshold)

            find_anchors += anchors.tolist()

            for i in anchors:
                y[i] = [0] + classes + [cx, cy, w, h]

    if len(find_anchors) > 0 or unmatch_return:
        return y
    raise RuntimeError("There are no bounding boxes match")


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
    raise RuntimeError('No bounding boxes in new image')


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


def normalize(img):
    return (img - np.mean(img)) / (np.max(img) - np.min(img) + 1e-15)


def pre_process(img, y):
    img = np.expand_dims(img, axis=0)
    y = np.expand_dims(y, axis=0)

    return normalize(img), y


def original_image(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        return pre_process(img, y)
    except RuntimeError:
        return None


def flip_n_saturation(img, bboxes_raw):
    try:
        img_flip_h = np.flip(img, 1)
        img_s = tf.image.random_saturation(img_flip_h, .4, 1.6)

        bboxes = deepcopy(bboxes_raw)
        bboxes[:, -4] = 1 - bboxes[:, -4]
        y = match_bbox(bboxes)

        return pre_process(img_s, y)
    except RuntimeError:
        return None


def flip_horiz(img, bboxes_raw):
    try:
        img_flip_h = np.flip(img, 1)

        bboxes = deepcopy(bboxes_raw)
        bboxes[:, -4] = 1 - bboxes[:, -4]
        y = match_bbox(bboxes)

        return pre_process(img_flip_h, y)
    except RuntimeError:
        return None


def saturation(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        img_s = tf.image.random_saturation(img, .4, 1.6)
        return pre_process(img_s, y)
    except RuntimeError:
        return None


def contrast(img, bboxes_raw):
    try:
        bboxes = deepcopy(bboxes_raw)
        y = match_bbox(bboxes)
        img_c = tf.image.random_contrast(img, .5, .6)
        return pre_process(img_c, y)
    except RuntimeError:
        return None


def zoom(img, bboxes_raw, proportion=.7, delta_x=0, delta_y=0):
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, proportion, delta_x, delta_y)

        y = match_bbox(bboxes)

        return pre_process(img_z, y)
    except RuntimeError:
        return None


def augumentation_level_0(img, bboxes):
    return [original_image(img, bboxes)]


def augumentation_level_1(img, bboxes):
    result = augumentation_level_0(img, bboxes)
    result += [flip_horiz(img, bboxes)]
    return result


def augumentation_level_2(img, bboxes):
    result = augumentation_level_1(img, bboxes)
    result += [saturation(img, bboxes)]
    return result


def augumentation_level_3(img, bboxes):
    result = augumentation_level_2(img, bboxes)
    result += [contrast(img, bboxes),
               flip_n_saturation(img, bboxes)]
    return result


def augumentation_level_4(img, bboxes):
    result = augumentation_level_3(img, bboxes)
    result += [zoom(img, bboxes, .8, -.099,  .099),
               zoom(img, bboxes, .8,     0,  .099),
               zoom(img, bboxes, .8,  .099,  .099),
               zoom(img, bboxes, .8, -.099,     0),
               zoom(img, bboxes, .8,     0,     0),
               zoom(img, bboxes, .8,  .099,     0),
               zoom(img, bboxes, .8, -.099, -.099),
               zoom(img, bboxes, .8,     0, -.099),
               zoom(img, bboxes, .8,  .099, -.099)]
    return result


def augumentation_level_5(img, bboxes):
    result = augumentation_level_4(img, bboxes)
    result += [zoom(img, bboxes, .6, -.2,  .2),
               zoom(img, bboxes, .6,   0,  .2),
               zoom(img, bboxes, .6,  .2,  .2),
               zoom(img, bboxes, .6, -.2,   0),
               zoom(img, bboxes, .6,   0,   0),
               zoom(img, bboxes, .6,  .2,   0),
               zoom(img, bboxes, .6, -.2, -.2),
               zoom(img, bboxes, .6,   0, -.2),
               zoom(img, bboxes, .6,  .2, -.2),
               contrast(img, bboxes),
               flip_n_saturation(img, bboxes)]
    return result



def data_augmentation(image_info, bboxes):
    img_bin = image.load_img('project/' + image_info[1], target_size=(300, 300))
    img = image.img_to_array(img_bin)

    img_categories = OHC.inverse_transform(bboxes[:, :-4]).flatten()

    aug_level = np.median([AUG_CRITERIA[c] for c in img_categories])

    if aug_level == 0:
        results = augumentation_level_0(img, bboxes)
    elif aug_level <= 1:
        results = augumentation_level_1(img, bboxes)
    elif aug_level <= 2:
        results = augumentation_level_2(img, bboxes)
    elif aug_level <= 3:
        results = augumentation_level_3(img, bboxes)
    elif aug_level <= 4:
        results = augumentation_level_4(img, bboxes)
    else:
        results = augumentation_level_5(img, bboxes)

    return list(filter(partial(is_not, None), results))
