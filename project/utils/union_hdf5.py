# imports
import os

import modin.pandas as pd

import project.download_content as content

from tqdm import tqdm


def main():
    tqdm.pandas()
    modelpath = os.path.join(content.DATAPATH, "MODEL")

    df = pd.concat([pd.read_hdf(os.path.join(modelpath, f"data_preprocessed_{i}.h5"),
                                'X', mode='r') for i in range(8)])

    filepath = os.path.join(modelpath, "data_preprocessed.h5")

    df.to_hdf(filepath, key='X', mode='w')


if __name__ == '__main__':
    main()
