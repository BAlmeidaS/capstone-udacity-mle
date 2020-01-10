# imports
import os

import numpy as np
import pandas as pd
import json

from time import sleep, time

from project.model.ssd_model_300_mobilenet import ssd_model_300_mobilenet
from project.model.ssd_model_300_xception import ssd_model_300_xception
from project.model.ssd_model_300_resnet import ssd_model_300_resnet
from project.model.ssd_model_300_vgg import ssd_model_300_vgg

from project.utils.data_bbox_match_hdf5 import load_ohc

import h5py

from tqdm import trange

import project.download_content as content


from project.model.iou import iou

from sklearn.metrics import auc

METAPATH = os.path.join(content.DATAPATH, 'METADATA')
MODELPATH = os.path.join(content.DATAPATH, 'MODEL')

CROSS_VAL_PREPROCESSED = os.path.join(content.DATAPATH,
                                      "MODEL",
                                      "cross_val_data_preprocessed.h5")

TEST_PREPROCESSED = os.path.join(content.DATAPATH,
                                 "MODEL",
                                 "test_data_preprocessed.h5")
ohc = load_ohc()


class Evaluator():
    def __init__(self, model_ref, model_fn, weights, df_ref):
        model = model_fn(inference=True)
        model.load_weights(weights)

        self.model = model
        self.model_ref = model_ref
        self.df_ref = df_ref

    def load_images(self, label_id):
        images = []

        with h5py.File(os.path.join(MODELPATH, 'cross_val', f'{label_id}.h5'), mode='r') as f:
            images_ids = list(f.keys())
            for k in images_ids:
                images.append(f[k][:])

        return np.concatenate(images, axis=0), images_ids

    def match_predictions(self, label_id, predictions, image_ids):
        results = []

        for img_id, img_pred in zip(image_ids, predictions):
            ground_truths = (self.df_ref[(self.df_ref.ImageID == img_id)
                                         & (self.df_ref.LabelID == label_id)]
                             [['cx', 'cy', 'w', 'h']]
                             .values)

            bboxes_preds = np.pad(img_pred[img_pred[..., 0] == label_id],
                                  [[0, 0], [0, 1]])

            if bboxes_preds.shape[0] > 0:
                for bbox_label in ground_truths:
                    ious = iou(bbox_label, bboxes_preds[..., -5:-1])
                    index = np.argmax(ious)

                    if ious[index] >= .5:
                        bboxes_preds[index, -1] = 1

            results.append(bboxes_preds)

        return np.concatenate(results)

    def calc_auc(self, label_id, total_bbox, matched):
        true_positives = matched[matched[..., -1] == 1][..., 1]

        if true_positives.shape[0] == 0:
            return 0

        percs = [np.percentile(true_positives, i) for i in range(0, 100, 3)]

        precisions = []
        recalls = []

        for perc in percs:
            if matched[matched[:, 1] > perc].shape[0] > 0:
                precision = (true_positives[true_positives > perc].shape[0]
                             / matched[matched[:, 1] > perc].shape[0])
            else:
                precision = 0

            recall = (true_positives[true_positives > perc].shape[0]
                      / total_bbox)

            precisions.append(precision)
            recalls.append(recall)

        # ordering recall and replicate to precisions
        sort_p, sort_r = zip(*sorted(zip(precisions, recalls),
                                     key=lambda x: x[1]))

        return auc(sort_r, sort_p)

    def eval_category(self, label_id):
        label = ohc.get_feature_names()[label_id-1][3:]
        imgs, img_ids = self.load_images(label_id)
        preds = self.model.predict(imgs)
        matched = self.match_predictions(label_id, preds, img_ids)

        total_bbox = self.df_ref[self.df_ref.LabelID == label_id].shape[0]

        return label, self.calc_auc(label_id, total_bbox, matched)

    def eval_model(self):
        total = {}

        print(f'EVALUATING {self.model_ref.upper()}...')
        sleep(.5)

        # measuring time elapsed
        start = time()

        for i in trange(0, 600):
            try:
                # evaluation AP for each category
                ap = self.eval_category(i)
                total[ap[0]] = ap[1]
            except OSError:
                pass
        # ending measure
        end = time()

        total['time_elapsed'] = f'{end - start:.0f} seconds'

        with open(os.path.join(MODELPATH, f'AP-{self.model_ref}.json'), 'w') as f:
            json.dump(total, f)


def main(bbox_path):
    modelrefs = {
        "xception-v2-testdataset": {
            "model_fn": ssd_model_300_xception,
            "weights": os.path.join(MODELPATH, "v2/xception-weights300-v2.h5"),
        }
    }

    bboxes = pd.read_hdf(bbox_path, key='X', mode='r')

    for k, v in modelrefs.items():
        model_eval = Evaluator(k, v['model_fn'], v['weights'], bboxes)
        model_eval.eval_model()


if __name__ == '__main__':
    main(TEST_PREPROCESSED)
