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

import logging
logging.getLogger().setLevel(logging.INFO)


def lr_schedule_builder(epochs):
    step = int(epochs/3)

    def lr_schedule(epoch):
        if epoch < step:
            return 1e-3
        elif epoch < 2*step:
            return 1e-4
        else:
            return 1e-5

    return lr_schedule


def load_model(fn):
    model = fn()

    opt = SGD(learning_rate=1, momentum=0.9, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


def main(model_type, model_fn, batch_size=20, steps_per_epoch=200, batch_images=150):
    model = load_model(model_fn)
    model.summary()

    # model.load_weights(content.DATAPATH + '/0to3-full-weights300vgg16.h5')

    def gen_data():
        # each database with data preprocessed
        files = [
            "/media/external/39_classes_300x300_3.h5",
            "/media/external/39_classes_300x300_6.h5",
            "/media/external/39_classes_300x300_7.h5",
            "/media/external/39_classes_300x300_0.h5",
            "/media/external/39_classes_300x300_2.h5",
            "/media/external/39_classes_300x300_5.h5",
            "/media/external/39_classes_300x300_1.h5",
            "/media/external/39_classes_300x300_4.h5",
        ]

        # this while true is to avoid keras error with generator functions
        while True:
            # X and y are preloading data for X and y (must be greater than
            # batch size
            X = None
            y = None

            # iterate in each file
            for f_path in files[:4]:
                f = h5py.File(f_path, 'r')
                batches = f['batches'][:]
                np.random.shuffle(batches)

                try:
                    for x_ref, y_ref in batches:
                        if X is None and y is None:
                            # preload the first data ref
                            X = f[x_ref][:]
                            y = f[y_ref][:]
                        else:
                            # appending new X in old X
                            X = np.concatenate([X, f[x_ref][:]], axis=0)
                            y = np.concatenate([y, f[y_ref][:]], axis=0)

                        while X.shape[0] > batch_size:
                            # getting the first items in each tensor
                            batch_x = X[:batch_size]
                            batch_y = y[:batch_size]

                            # reduncing tensor size of axis -1
                            X = X[batch_size:]
                            y = y[batch_size:]

                            # yielding batch_x and batch_y to network training
                            yield batch_x, batch_y
                finally:
                    f.close()

    # hdf5 handle notebook explain this number
    total_images = 2370854
    epochs = int(total_images / (batch_size * steps_per_epoch)) + 1

    lr_schedule = lr_schedule_builder(epochs)
    lr_callback = LearningRateScheduler(lr_schedule)

    callbacks = [lr_callback]

    model.fit_generator(gen_data(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=0)

    model.save_weights(content.DATAPATH + f'/{model_type}-weights300vgg16.h5')


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

    main(model, fn)
