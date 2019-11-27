import h5py
import pandas as pd
import numpy as np
import os

# from project.utils import configs

from keras.preprocessing import image
from keras.optimizers import Adam

import project.download_content as content
from project.model.ssd_model_300 import ssd_model_300
from project.model.loss import SSDloss

import logging
logging.getLogger().setLevel(logging.INFO)


def load_data():
    filepath = os.path.join(content.DATAPATH, "MODEL", "boats_300_vgg.h5")

    boats = pd.read_hdf(filepath, 'X', mode='r')

    with h5py.File(filepath, 'r') as f:
        target = f['y'][:]

    return boats[['ImageID', 'Path']], target


def load_model():
    model = ssd_model_300()

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # opt = SGD(learning_rate=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    ssd_loss = SSDloss()
    model.compile(optimizer=opt, loss=ssd_loss.loss)

    return model


def main():
    X, target = load_data()

    model = load_model()
    model.summary()

    def gen_data():
        while True:
            for i, y in enumerate([target[23000]]):
                i=22000
                img_path = X.loc[i].Path
                img = image.load_img('project/' + img_path, target_size=(300, 300))
                img_arr = image.img_to_array(img)

                img_arr = np.expand_dims(img_arr, axis=0)
                y = np.expand_dims(y, axis=0)

                yield img_arr, y

    model.fit_generator(gen_data(), steps_per_epoch=128, epochs=40, workers=0)

    model.save_weights(content.DATAPATH + '/weights300vgg16.h5')


if __name__ == '__main__':
    main()
