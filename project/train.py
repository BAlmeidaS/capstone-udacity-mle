import h5py
import numpy as np
import tensorflow as tf
import os

from copy import deepcopy

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
    return groups


def load_data(group):
    """return an array with images infos"""
    with h5py.File(DATAPATH, 'r') as f:
        images = f['0']['images'][:]
        X = np.array([f['0'][i[0]][:] for i in images])

    return images, X


def count_images():
    groups = get_group_imgs()

    count_images = 0

    with h5py.File(DATAPATH, 'r') as f:
        for group in groups:
            count_images += f[group]['images'][:].shape[0]

    return count_images


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


def data_augmentation(image_info, bboxes_raw):
    img_bin = image.load_img('project/' + image_info[1], target_size=(300, 300))
    img = image.img_to_array(img_bin)

    bboxes = deepcopy(bboxes_raw)
    y = match_bbox(bboxes)
    yield img, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]

    # flip horizontaly
    img_flip_h = np.flip(img, 1)

    bboxes = deepcopy(bboxes_raw)
    bboxes[:, -4] = 1 - bboxes[:, -4]
    y = match_bbox(bboxes)

    yield img_flip_h, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]

    # flip verticaly
    img_flip_v = np.flip(img, 0)

    bboxes = deepcopy(bboxes_raw)
    bboxes[:, -3] = 1 - bboxes[:, -3]
    y = match_bbox(bboxes)

    yield img_flip_v, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]

    # flip vertically and horizontaly
    img_flip_h_v = np.flip(img, [0, 1])

    bboxes = deepcopy(bboxes_raw)
    bboxes[:, -4:-2] = 1 - bboxes[:, -4:-2]
    y = match_bbox(bboxes)

    yield img_flip_h_v, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]

    # zoom in center
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .84, .079, .079)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in center 2
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .7, .149, .149)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in center 3
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .6, 0, 0)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in top left
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .8, -0.09, -.099)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in top right
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .8, .099, -.099)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in bottom left
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .8, -.099, .099)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in bottom right
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .8, .099, .099)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in top left 2
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .6, -.2, -.2)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in top right 2
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .6, .2, -.2)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in bottom left 2
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .6, -.2, .2)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass

    # zoom in bottom right 2
    try:
        bboxes = deepcopy(bboxes_raw)
        img_z, bboxes = resize(img, bboxes, .6, .2, .2)

        y = match_bbox(bboxes)

        yield img_z, y[:, [0, 53, 301, 465, -4, -3, -2, -1]]
    except ValueError:
        pass


def main(batch_size=16, steps_per_epoch=128):
    model = load_model()
    model.summary()

    def gen_data():
        for group in get_group_imgs():
            images, X = load_data(group)

            for image_info, bboxes in zip(images, X):
                data_generator = data_augmentation(image_info, bboxes)

                while True:
                    try:
                        img, y = next(data_generator)

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
    data_aug_empirical = 10
    # epochs = (num_images * data aug)/(steps_per_epoch * batch_size)
    model.fit_generator(batch_gen_data(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=int((count_images() * data_aug_empirical)
                                   / (steps_per_epoch * batch_size) + 1),
                        workers=0)

    model.save_weights(content.DATAPATH + '/weights300vgg16.h5')


if __name__ == '__main__':
    # main(batch_size=1, steps_per_epoch=50)
    main()
