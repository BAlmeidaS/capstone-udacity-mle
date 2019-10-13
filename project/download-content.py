"""
This module is responsible to download all type of files to run the project.

Example:
    $ python download-dataset.py
    ...
"""

import os
import requests
import shutil

from tqdm import tqdm

DATASETS_URLS = ['https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-detection-human-imagelabels.csv',
                 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-detection-bbox.csv',
                 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-detection-human-imagelabels.csv',
                 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-detection-bbox.csv',
                 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-500.csv',
                 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label500-hierarchy.json']

ROOTPATH = os.path.dirname(os.path.relpath(__file__))


def create_data_folder(folder: str):
    """create_data_folder

    Args:
        folder (str): folder to be created

    Raises:
        SystemExit: If user opts to not delete folder
    """
    path = os.path.join(ROOTPATH, 'data', folder)

    if os.path.exists(path):
        if get_input(f'{path} ALREADY EXIST!\nDo you want to delete? '):
            print(f'deleting {path}...')
            shutil.rmtree(path)
        else:
            raise SystemExit('Exit by user!')

    print(f'creating {path}...\n')
    os.mkdir(path)


def get_filename(url: str):
    """Get filename from content-disposition

    Args:
        url (str): the url
    """
    return url[url.rfind("/") + 1:]


def download(url: str, folder: str):
    """Function that Download somethi

    Args:
        url (str): the url for download file
        folder (str): the relative path of folder where to write the file downloaded
    """
    # Streaming, so we can iterate over the response.
    req = requests.get(url, stream=True, allow_redirects=True)

    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))

    # define the block size
    block_size = 1024

    # the path for save file - using data folder
    path = os.path.join(ROOTPATH, 'data', folder)

    # the filename get from request
    filename = get_filename(url)

    fullpath = os.path.join(path, filename)

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(fullpath, 'wb') as f:
        for data in req.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        RuntimeError(f"ERROR DURING DOWNLOAD {url}")


def download_datasets():
    """Download all datasets"""
    print("Starting datasets downloads...\n")
    for url in DATASETS_URLS:
        download(url, 'datasets')


def download_images():
    """Download all images"""
    print("Starting images downloads...\n")


def get_input(msg: str) -> bool:
    """Handle with user input.

    Args:
        msg: Message to show in input command

    Returns:
        bool: True if user inputs yes, else False

    Raises:
        ValueError: If users types other thing than 'y', 'n', 'yes' or 'no'
                    case-insensitive.
    """
    positives = ['yes', 'y']
    negatives = ['no', 'n']

    choice = str(input(msg)).lower()

    if choice not in (positives + negatives):
        raise ValueError('NOT A VALID OPTION!')
    if choice in positives:
        return True
    else:
        return False


if __name__ == '__main__':
    print('Answer Y(yes) or N(No)')

    # input if user wants to download metadata
    datasets = get_input('Do you want download metadata about images (only datasets)?: ')
    # create folder if need it
    if datasets:
        create_data_folder('datasets')

    # input if user wants to download images
    images = get_input('Do you want download ALL images (this could take longer)? : ')
    # create folder if need it
    if images:
        create_data_folder('images')

    # start downloads
    if datasets:
        download_datasets()
    if images:
        download_images()

    print('All done! =]')
