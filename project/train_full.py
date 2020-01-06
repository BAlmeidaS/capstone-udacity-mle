import os

import h5py
import numpy as np

import argparse

from keras.optimizers import SGD
from keras.callbacks.callbacks import LearningRateScheduler

import project.download_content as content
from project.model.ssd_model_300_vgg import ssd_model_300_vgg
from project.model.ssd_model_300_resnet import ssd_model_300_resnet
from project.model.ssd_model_300_xception import ssd_model_300_xception
from project.model.loss import SSDloss

import ray

import logging
logging.getLogger().setLevel(logging.INFO)

METAPATH = os.path.join(content.DATAPATH, 'METADATA')
TRAINPATH = os.path.join(METAPATH, 'train_data_650.h5')


def lr_schedule(epoch):
    if epoch < 15:
        return 1e-3
    elif epoch < 25:
        return 1e-4
    else:
        return 1e-5


def load_model(fn):
    model = fn()

    opt = SGD(learning_rate=1, momentum=0.9, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


@ray.remote
def get_item(x_ref, y_ref):
    with h5py.File(TRAINPATH, 'r') as f_temp:
        X = f_temp[x_ref][:]
        y = f_temp[y_ref][:]
    return X, y


def main(model_type, model_fn, batch_size=20, steps_per_epoch=8130, epochs=5):
    model = load_model(model_fn)
    model.summary()

    # model.load_weights(content.DATAPATH + '/0to3-full-weights300vgg16.h5')

    def gen_data():
        while True:
            with h5py.File(TRAINPATH, 'r') as f_temp:
                batches = f_temp['batches'][:]
                np.random.shuffle(batches)

            X = None
            y = None

            # iterate in each file
            results = []
            for x_ref, y_ref in batches:
                results.append(get_item.remote(x_ref, y_ref))

                if len(results) > 5:
                    while len(results) > 0:
                        done_id, results = ray.wait(results)

                        if X is None and y is None:
                            X, y = ray.get(done_id[0])
                        else:
                            X_t, y_t = ray.get(done_id[0])
                            X = np.concatenate([X, X_t], axis=0)
                            y = np.concatenate([y, y_t], axis=0)

                        while X.shape[0] > batch_size:
                            # getting the first items in each tensor
                            batch_x = X[:batch_size]
                            batch_y = y[:batch_size]

                            # reduncing tensor size of axis -1
                            X = X[batch_size:]
                            y = y[batch_size:]

                            # yielding batch_x and batch_y to network training
                            yield batch_x, batch_y

    lr_callback = LearningRateScheduler(lr_schedule)

    callbacks = [lr_callback]

    ray.init(num_cpus=6)
    try:
        model.fit_generator(gen_data(),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks,
                            workers=0)
    finally:
        ray.shutdown()

    model.save_weights(content.DATAPATH + f'/{model_type}-weights300.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running a training')
    parser.add_argument('base_model', type=str,
                        help='choose the base model to train ssd',
                        choices=['vgg', 'resnet', 'xception'])
    model = parser.parse_args().base_model

    if model == 'xception':
        fn = ssd_model_300_xception
    elif model == 'resnet':
        fn = ssd_model_300_resnet
    else:
        fn = ssd_model_300_vgg

    main(model, fn, batch_size=20, steps_per_epoch=1837, epochs=30)
