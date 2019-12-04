import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# from project.utils import configs

from keras.preprocessing import image
from keras.optimizers import SGD
from keras.optimizers import Adam

import project.download_content as content
from project.model.ssd_model_300 import ssd_model_300
# from project.model.ssd_model_624_vgg_19 import ssd_model_624_vgg_19
from project.model.loss import SSDloss, BBOX_REF

import logging
logging.getLogger().setLevel(logging.INFO)


def load_data():
    filepath = os.path.join(content.DATAPATH, "MODEL", "part_data_300_vgg.h5")

    X = pd.read_hdf(filepath, 'X', mode='r')

    with h5py.File(filepath, 'r') as f:
        target = f['y'][:]

    return X, target


def load_model():
    model = ssd_model_300()

    # opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    opt = SGD(learning_rate=1e-5, momentum=0.5, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


def match_bbox(x, y):
    bboxes_x = []
    for cx, cy, w, h, *labels in zip(x.cx, x.cy, x.w, x.h, *x[8:]):
        bboxes = BBOX_REF.match(pd.DataFrame({'cx': [cx],
                                              'cy': [cy],
                                              'w': [w],
                                              'h': [h]}).loc[0])
        bboxes_x.append(bboxes)
        for i in bboxes:
            y[i] = [0] + labels + [cx, cy, w, h]

    for match_bboxes in bboxes_x:
        if len(match_bboxes) > 0:
            x.bbox_ref = bboxes_x
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


def data_augmentation(ind, X, y):
    x = X.iloc[ind].copy()

    img_path = x.Path
    img_b = image.load_img('project/' + img_path, target_size=(300, 300))

    img = image.img_to_array(img_b)

    yield img, y

    # flip horizontaly
    y[:] = [1] + [0] * (y.shape[1] - 1)

    img_flip_h = np.flip(img, 1)
    x.cx = np.subtract(1, x.cx)
    y = match_bbox(x, y)

    yield img_flip_h, y

    # flip verticaly
    y[:] = [1] + [0] * (y.shape[1] - 1)

    img_flip_v = np.flip(img, 0)
    x.cy = np.subtract(1, x.cy)
    y = match_bbox(x, y)

    yield img_flip_v, y

    # flip vertically and horizontaly
    y[:] = [1] + [0] * (y.shape[1] - 1)

    img_flip_h_v = np.flip(img, [0, 1])
    x.cx = np.subtract(1, x.cx)
    x.cy = np.subtract(1, x.cy)
    y = match_bbox(x, y)

    yield img_flip_h_v, y

    # zoom in center
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .84, .079, .079)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in center 2
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .7, .149, .149)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in center 3
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .6, 0, 0)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in top left
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .8, -0.09, -.099)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in top right
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .8, .099, -.099)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in bottom left
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .8, -.099, .099)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in bottom right
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .8, .099, .099)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in top left 2
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .6, -.2, -.2)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in top right 2
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .6, .2, -.2)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in bottom left 2
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .6, -.2, .2)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass

    # zoom in bottom right 2
    try:
        y[:] = [1] + [0] * (y.shape[1] - 1)

        img_z, bboxes = resize(img, x, .6, .2, .2)
        x.cx = bboxes[:, 0]
        x.cy = bboxes[:, 1]
        x.w = bboxes[:, 2]
        x.h = bboxes[:, 3]
        for i, r in enumerate(bboxes[:, 4:].T):
            x[8+i] = r
        y = match_bbox(x, y)

        yield img_z, y
    except ValueError:
        pass


def main():
    X, target = load_data()

    model = load_model()
    model.summary()

    def gen_data():
        while True:
            for ind, y in enumerate(target):
                data_generator = data_augmentation(ind, X, y)

                while True:
                    try:
                        img, y = next(data_generator)

                        img = np.expand_dims(img, axis=0)
                        y = np.expand_dims(y, axis=0)

                        yield ((img - np.mean(img)) / (np.std(img) + 1e-15)), y
                    except StopIteration:
                        break

    def batch_gen_data():
        while True:
            batch_x, batch_y = None, None
            batch_size = 16

            for i, (x, y) in enumerate(gen_data()):
                if i % batch_size == 0:
                    if batch_x is not None and batch_y is not None:
                        yield batch_x, batch_y

                    batch_x = x
                    batch_y = y
                    continue

                batch_x = np.concatenate([batch_x, x], axis=0)
                batch_y = np.concatenate([batch_y, y], axis=0)

    model.fit_generator(batch_gen_data(),
                        steps_per_epoch=128,
                        epochs=240,
                        workers=0)

    model.save_weights(content.DATAPATH + '/weights300vgg16.h5')


if __name__ == '__main__':
    main()
