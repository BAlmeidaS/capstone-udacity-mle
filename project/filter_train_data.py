import os

import numpy as np
import h5py

import json
from tqdm import tqdm

import project.download_content as content

METAPATH = os.path.join(content.DATAPATH, 'METADATA')


def main():
    with open(os.path.join(METAPATH, 'filtered_data.json')) as f:
        filter_criteria = json.load(f)

    final_file = h5py.File(os.path.join(METAPATH, 'train_data.h5'), 'w')

    try:
        for filename in filter_criteria.keys():
            path = os.path.join('/media/external', filename + '.h5')

            with h5py.File(path, 'r') as f_temp:
                for batch in tqdm(filter_criteria[filename]):
                    x = f_temp[batch + '-x'][:]
                    y = f_temp[batch + '-y'][:]

                    final_file.create_dataset(name=f"{batch}-x",
                                              data=x,
                                              dtype=np.float16,
                                              compression='gzip',
                                              compression_opts=1)

                    final_file.create_dataset(name=f"{batch}-y",
                                              data=y,
                                              dtype=np.float16,
                                              compression='gzip',
                                              compression_opts=1)
        keys = final_file.keys()

        unique_ids = set(i[:-2] for i in keys)

        batches = []
        for id in unique_ids:
            batches.append([f"{id}-x".encode("ascii", "ignore"),
                            f"{id}-y".encode("ascii", "ignore")])

        final_file.create_dataset(name='batches',
                                  shape=(len(batches), 2),
                                  data=batches,
                                  dtype=h5py.special_dtype(vlen=str),
                                  compression='gzip',
                                  compression_opts=1)

    finally:
        final_file.close()


if __name__ == '__main__':
    main()
