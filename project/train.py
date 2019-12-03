import h5py
import pandas as pd
import numpy as np
import os

# from project.utils import configs

from keras.preprocessing import image
from keras.optimizers import SGD
from keras.optimizers import Adam

import project.download_content as content
from project.model.ssd_model_300 import ssd_model_300
# from project.model.ssd_model_624_vgg_19 import ssd_model_624_vgg_19
from project.model.loss import SSDloss

import logging
logging.getLogger().setLevel(logging.INFO)


def load_data():
    filepath = os.path.join(content.DATAPATH, "MODEL", "part_data_300_vgg.h5")

    X = pd.read_hdf(filepath, 'X', mode='r')

    with h5py.File(filepath, 'r') as f:
        target = f['y'][:]

    return X[['ImageID', 'Path']], target


def load_model():
    model = ssd_model_300()

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # opt = SGD(learning_rate=1e10-3, momentum=0.9, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


def main():
    X, target = load_data()

    model = load_model()
    model.summary()

    def gen_data():
        while True:
            for ind, y in enumerate(target):
                img_path = X.iloc[ind].Path
                img = image.load_img('project/' + img_path, target_size=(300, 300))
                img_arr = image.img_to_array(img)

                img_arr = np.expand_dims(img_arr, axis=0)
                y = np.expand_dims(y, axis=0)

                yield img_arr, y

    def batch_gen_data():
        while True:
            batch_x, batch_y = None, None
            batch_size = 8

            for i, (x, y) in enumerate(gen_data()):
                if i % batch_size == 0:
                    if batch_x is not None and batch_y is not None:
                        import ipdb; ipdb.set_trace()
                        yield batch_x, batch_y

                    batch_x = x
                    batch_y = y
                    continue

                batch_x = np.concatenate([batch_x, x], axis=0)
                batch_y = np.concatenate([batch_y, y], axis=0)


    model.fit_generator(batch_gen_data(),
                        steps_per_epoch=128,
                        epochs=40,
                        workers=0)

    model.save_weights(content.DATAPATH + '/weights300vgg16.h5')


if __name__ == '__main__':
    main()
