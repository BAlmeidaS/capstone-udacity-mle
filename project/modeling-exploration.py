#!/usr/bin/env python
# coding: utf-8

import os
os.sys.path.append(os.path.abspath(".."))

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from notebooks_utils import visuals

import project.download_content as content

from keras.preprocessing import image
from keras.optimizers import Adam

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import multiprocessing

from project.model.ssd_model_300 import ssd_model_300
from project.model.loss import SSDloss
from project.model.iou import iou


METAPATH = os.path.join(content.DATAPATH, 'METADATA')

df = pd.read_csv(METAPATH + '/enriched_train_bbox.csv')
boats = df.query("LabelSemantic == 'Boat'")

model = ssd_model_300()
model.summary()

sgd = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
ssd_loss = SSDloss()
model.compile(optimizer=sgd, loss=ssd_loss.loss)


def bbox(sizes, stride=1, ratios=None):
    arr = []
    for n in sizes:
        e = 1 / n

        t = e*3

        for i in range(0, n-2, stride):
            for j in range(0, n-2, stride):
                arr.append([e*(i+3/2), e*(j+3/2), t, t])

                for r in ratios:
                    arr.append([e*(i+3/2), e*(j+3/2), t*r, t])
                for r in ratios:
                    arr.append([e*(i+3/2), e*(j+3/2), t, t*r])
    return pd.DataFrame(arr, columns=['cx', 'cy', 'w', 'h'])


refs_s2 = bbox([75, 39], stride=2, ratios=[1/2])
refs_s1 = bbox([11, 7, 5, 3], stride=1, ratios=[1/2, 1/3])
refs = pd.concat([refs_s2, refs_s1]).reset_index()[['cx', 'cy', 'w', 'h']]


def identify_bbox(row):
    row = row[['cx', 'cy', 'w', 'h']]
    ious = iou(row.values, refs.values)
    ious[ious < 0.5] = 0

    bboxs = np.where((ious == np.amax(ious)) & (ious > 0))[0]
    return bboxs


with ProgressBar():
    boats['bbox_ref'] = (dd.from_pandas(boats, npartitions=multiprocessing.cpu_count())
                           .map_partitions(lambda df: df.apply(identify_bbox, axis=1))
                           .compute(scheduler='processes'))


imgs = boats[['ImageID', 'Path']].drop_duplicates()
imgs.set_index('ImageID', inplace=True)

for c in ['cx', 'cy', 'w', 'h', 'bbox_ref']:
    imgs = imgs.join(boats.groupby('ImageID')[c].apply(list))

imgs.reset_index(inplace=True)


target = np.zeros((imgs.shape[0], refs.shape[0], model.output.shape[-1]), dtype=np.float32)

target[:][:] = [1, 0, 0, 0, 0, 0]

for i, r in imgs.iterrows():
    for cx, cy, w, h, refs in zip(r.cx, r.cy, r.w, r.h, r.bbox_ref):
        for ref in refs:
            target[i][int(ref)] = [0, 1, cx, cy, w, h]


def gen_data():
    while True:
        for i, y in enumerate(target):
            img_path = imgs.iloc[i].Path
            img = image.load_img(img_path, target_size=(300, 300))
            X = image.img_to_array(img)

            X = np.expand_dims(X, axis=0)
            y = np.expand_dims(y, axis=0)

            yield X, y


model.fit_generator(gen_data(), steps_per_epoch=4, epochs=6500)


img_path = imgs.iloc[421].Path
img = image.load_img(img_path, target_size=(300, 300))
X = image.img_to_array(img)

p = model.predict(np.expand_dims(X, axis=0))


arr = [x for x in p[0] if x[1] > .99]
df_arr = pd.DataFrame(arr, columns=['no-class', 'boat', 'cx', 'cy', 'w', 'h'])
df_arr.shape

plt.figure(figsize=(12, 8))
plt.imshow(img)

for bbox_found in df_arr.itertuples():
    visuals.draw_bbox(img, bbox_found, legend=False, color="#ff0000")
