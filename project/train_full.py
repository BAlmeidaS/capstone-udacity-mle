import h5py
import numpy as np
import os

from project.utils import data_augmentation as da

from keras.optimizers import SGD

import project.download_content as content
from project.model.ssd_model_300_full import ssd_model_300
from project.model.loss import SSDloss, BBOX_REF

import logging
logging.getLogger().setLevel(logging.INFO)

DATAPATH = os.path.join(content.DATAPATH, "MODEL", "data_300_vgg.h5")

ALL_ANCHORS = BBOX_REF.references.values


def count_images():
    groups = da.get_group_imgs()

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


def main(batch_size=16, steps_per_epoch=128):
    model = load_model()
    model.summary()

    def gen_data():
        for group in da.get_group_imgs():
            images, X = da.load_data(group)

            for image_info, bboxes in zip(images, X):
                data_generator = da.data_augmentation(image_info, bboxes)

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
