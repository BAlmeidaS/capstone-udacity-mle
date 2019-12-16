import h5py
import numpy as np
import ray
from collections import deque

from project.utils import data_augmentation as da

from keras.optimizers import SGD

import project.download_content as content
from project.model.ssd_model_300_full import ssd_model_300
from project.model.loss import SSDloss

import logging
logging.getLogger().setLevel(logging.INFO)


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


def main(batch_size=24, steps_per_epoch=128, batch_images=30):
    ray.init(ignore_reinit_error=True)

    try:
        model = load_model()
        model.summary()

        def gen_data():
            for group in da.get_group_imgs():
                images, X = da.load_data(group)

                batch = []
                deq = deque(maxlen=batch_images)

                while True:
                    for i, (image_info, bboxes) in enumerate(zip(images, X), 1):
                        deq.append((image_info, bboxes))
                        if i % batch_images == 0:
                            while len(batch) > batch_size:
                                topk = batch[:batch_size]
                                batch = batch[batch_size:]

                                imgs = [k[0] for k in topk]
                                ys = [k[1] for k in topk]
                                batch_x = np.concatenate(imgs, axis=0)
                                batch_y = np.concatenate(ys, axis=0)

                                yield batch_x, batch_y

                            futures = [da.async_data_augmentation.remote(i, b)
                                       for i, b in deq]
                            results = ray.get(futures)

                            batch += [item for sublist in results for item in sublist]

        # value of how many data augs are made over each image
        data_aug_empirical = 6
        # epochs = (num_images * data aug)/(steps_per_epoch * batch_size)
        model.fit_generator(gen_data(),
                            steps_per_epoch=steps_per_epoch,
                            epochs=int((count_images() * data_aug_empirical)
                                       / (steps_per_epoch * batch_size) + 1),
                            workers=0)

        model.save_weights(content.DATAPATH + '/fullweights300vgg16.h5')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    # main(batch_size=1, steps_per_epoch=50)
    main()
