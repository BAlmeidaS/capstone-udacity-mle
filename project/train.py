import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# from project.utils import configs

from keras.preprocessing import image
from keras.optimizers import SGD

import project.download_content as content
from project.model.ssd_model_300 import ssd_model_300
# from project.model.ssd_model_624_vgg_19 import ssd_model_624_vgg_19
from project.model.loss import SSDloss, BBOX_REF

import logging
logging.getLogger().setLevel(logging.INFO)

DATAPATH = os.path.join(content.DATAPATH, "MODEL", "part_data_300_vgg.h5")

ALL_ANCHORS = BBOX_REF.references.values


def get_group_imgs():
    with h5py.File(DATAPATH, 'r') as f:
        groups = list(f.keys())
    return groups[:-1]


def load_image_infos(group):
    """return an array with images infos"""
    with h5py.File(DATAPATH, 'r') as f:
        images = f[group]['images'][:]

    return images


def load_data(image_info):
    """ Load data, returning X and y for each id image """
    with h5py.File(DATAPATH, 'r') as f:
        y = f[image_info[0]][:]

    x = image.load_img(image_info[1])

    return x, y


def load_model():
    model = ssd_model_300()

    # opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    opt = SGD(learning_rate=1e-3, momentum=0.9, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


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


def resize(img, x, p=.7, delta_x=0, delta_y=0):
    lb = (1.0 - p) / 2.0
    hb = 1.0 - lb

    if delta_x > lb or delta_y > lb:
        raise RuntimeError("Delta X or Delta Y can't be greater than the half of (1 - proportion)")

    bboxes = find_bbox(x, p, delta_x, delta_y)

    return (tf.image.crop_and_resize(np.expand_dims(img, axis=0),
                                     [[delta_x + lb,
                                       delta_y + lb,
                                       delta_x + hb,
                                       delta_y + hb]],
                                     [0],
                                     [300, 300])[0],
            bboxes)


def resize_params(x, p=.7, delta_x=0.0, delta_y=0.0):
    cx_0, cy_0, w_0, h_0 = x

    step = (1.0 - p) / 2.0
    cx = (cx_0 - delta_y - step) / p
    cy = (cy_0 - delta_x - step) / p
    w = w_0 / p
    h = h_0 / p

    return [cx, cy, w, h]


def append_if_valid(bboxes, cx, cy, w, h, *classes):
    if cx > (1 - w/2) or cx < w/2 or w > 1 or w < 0:
        return
    if cy > (1 - h/2) or cy < h/2 or h > 1 or h < 0:
        return

    bboxes.append([cx, cy, w, h, *classes])


def find_bbox(x, p=.7, delta_x=0.0, delta_y=0.0):
    bboxes = []
    for cx, cy, w, h, *classes in zip(x.cx, x.cy, x.w, x.h, *x[8:]):
        n_cx, n_cy, n_w, n_h = resize_params([cx, cy, w, h], p, delta_x, delta_y)
        append_if_valid(bboxes, n_cx, n_cy, n_w, n_h, *classes)

    for ref_bboxes in bboxes:
        if len(ref_bboxes) > 0:
            return np.array(bboxes)
    raise ValueError('No bounding boxes in new image')


def data_augmentation(image_info, bboxes_raw):
    bboxes = bboxes_raw.copy()

    img_bin = image.load_img('project/' + image_info[1], target_size=(300, 300))
    img = image.img_to_array(img_bin)

    y = match_bbox(bboxes)
    yield img, y

#    # flip horizontaly
#    y[:] = [1] + [0] * (y.shape[1] - 1)
#
#    img_flip_h = np.flip(img, 1)
#    x.cx = np.subtract(1, x.cx)
#    y = match_bbox(x, y)
#
#    yield img_flip_h, y
#
#    # flip verticaly
#    y[:] = [1] + [0] * (y.shape[1] - 1)
#
#    img_flip_v = np.flip(img, 0)
#    x.cy = np.subtract(1, x.cy)
#    y = match_bbox(x, y)
#
#    yield img_flip_v, y
#
#    # flip vertically and horizontaly
#    y[:] = [1] + [0] * (y.shape[1] - 1)
#
#    img_flip_h_v = np.flip(img, [0, 1])
#    x.cx = np.subtract(1, x.cx)
#    x.cy = np.subtract(1, x.cy)
#    y = match_bbox(x, y)
#
#    yield img_flip_h_v, y
#
#    # zoom in center
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .84, .079, .079)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in center 2
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .7, .149, .149)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in center 3
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .6, 0, 0)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in top left
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .8, -0.09, -.099)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in top right
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .8, .099, -.099)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in bottom left
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .8, -.099, .099)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in bottom right
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .8, .099, .099)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in top left 2
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .6, -.2, -.2)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in top right 2
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .6, .2, -.2)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in bottom left 2
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .6, -.2, .2)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass
#
#    # zoom in bottom right 2
#    try:
#        y[:] = [1] + [0] * (y.shape[1] - 1)
#
#        img_z, bboxes = resize(img, x, .6, .2, .2)
#        x.cx = bboxes[:, 0]
#        x.cy = bboxes[:, 1]
#        x.w = bboxes[:, 2]
#        x.h = bboxes[:, 3]
#        for i, r in enumerate(bboxes[:, 4:].T):
#            x[8+i] = r
#        y = match_bbox(x, y)
#
#        yield img_z, y
#    except ValueError:
#        pass


def main(batch_size=16, steps_per_epoch=128):
    model = load_model()
    model.summary()

    def gen_data():
        for group in get_group_imgs():
            for image_info in load_image_infos(group):
                with h5py.File(DATAPATH, 'r') as f:
                    bboxes = f[group][image_info[0]][:]

            data_generator = data_augmentation(image_info, bboxes)

            while True:
                try:
                    img, y = next(data_generator)

                    print(img.shape)
                    print(y.shape)
                    img = np.expand_dims(img, axis=0)
                    y = np.expand_dims(y, axis=0)

                    # normalize pixels
                    yield ((img - np.mean(img)) / (np.std(img) + 1e-15)), y
                except StopIteration:
                    break

    def batch_gen_data():
        batch_x, batch_y = None, None

        while True:
            g = gen_data()

            for i, (x, y) in enumerate(g):
                if i % batch_size == 0:
                    if batch_x is not None and batch_y is not None:
                        yield batch_x, batch_y

                    batch_x = x
                    batch_y = y
                    continue

                batch_x = np.concatenate([batch_x, x], axis=0)
                batch_y = np.concatenate([batch_y, y], axis=0)

    # value of how many data augs are made over each image
    data_aug_empirical = 15
    # epochs = (num_images * data aug)/(steps_per_epoch * batch_size)
    model.fit_generator(gen_data(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=int((32000 * data_aug_empirical)
                                   / (steps_per_epoch * batch_size) + 1),
                        workers=0)

    model.save_weights(content.DATAPATH + '/weights300vgg16.h5')


if __name__ == '__main__':
    # main(batch_size=1, steps_per_epoch=50)
    main()
