import h5py
import numpy as np
import ray
from collections import deque
from itertools import zip_longest

from project.utils import data_augmentation as da

from keras.optimizers import SGD

import project.download_content as content
from project.model.ssd_model_300 import ssd_model_300
from project.model.loss import SSDloss

import logging
logging.getLogger().setLevel(logging.INFO)

from multiprocessing import cpu_count


def count_images():
    groups = da.get_group_imgs()

    count_images = 0

    with h5py.File(da.DATAPATH, 'r') as f:
        for group in groups:
            count_images += f[group]['images'][:].shape[0]

    return count_images


def load_model():
    model = ssd_model_300()

    opt = SGD(learning_rate=1e-3, momentum=0.9, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


def grouper(iterable, n, fillvalue=None):
    "https://docs.python.org/3/library/itertools.html#itertools-recipes"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def main(batch_size=20, steps_per_epoch=200, batch_images=150):
    ray.init()

    try:
        model = load_model()
        model.summary()

        def gen_data():
            for group in da.get_group_imgs():
                images, X = da.load_data(group)

                batch = []
                futures = []

                while True:
                    for i, items in enumerate(
                        grouper(zip(images, X), batch_images), 1
                    ):
                        futures += [da.batch_data_augmentation.remote(items)]
                        if i % cpu_count() == 0:

                            while len(futures):
                                done_id, futures = ray.wait(futures)
                                results = ray.get(done_id[0])

                                batch += [item for sublist in results for item in sublist]

                                while len(batch) > batch_size:
                                    topk = batch[:batch_size]
                                    batch = batch[batch_size:]

                                    imgs = [k[0] for k in topk]
                                    ys = [k[1] for k in topk]
                                    batch_x = np.concatenate(imgs, axis=0)
                                    batch_y = np.concatenate(ys, axis=0)

                                    yield batch_x, batch_y

        # value of how many data augs are made over each image
        data_aug_empirical = 5
        # epochs = (num_images * data aug)/(steps_per_epoch * batch_size)
        model.fit_generator(gen_data(),
                            steps_per_epoch=steps_per_epoch,
                            epochs=int((count_images() * data_aug_empirical)
                                       / (steps_per_epoch * batch_size) + 1),
                            workers=0)

        model.save_weights(content.DATAPATH + '/weights300vgg16.h5')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    # main(batch_size=1, steps_per_epoch=50)
    main()
