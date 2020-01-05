# imports
import os

import pandas as pd
import numpy as np

from tqdm import tqdm

import project.download_content as content
from project.model.loss import BBOX_REF

import ray

METAPATH = os.path.join(content.DATAPATH, 'METADATA')

anchors = BBOX_REF.references.values


@ray.remote
def union_npy(root, folder):
    df = (pd.DataFrame(columns=(['file', 'batch']
                                + [f"c{i}" for i in range(599)]
                                + [f"b{i}" for i in range(8732)]))
            .set_index(['file', 'batch']))

    folderpath = os.path.join(root, folder)

    for file in tqdm(os.listdir(folderpath)):
        result = np.load(os.path.join(folderpath, file))
        df.loc[(folder, file[:-4]), :] = result

    (df.reset_index()
       .to_csv(folderpath + '.csv', index=False))


def main():
    ray.init()
    try:
        futures = []
        root = os.path.join(METAPATH, 'dataaug')
        for folder in os.listdir(root):
            futures.append(union_npy.remote(root, folder))
        ray.get(futures)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
