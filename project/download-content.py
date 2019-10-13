"""
This module is responsible to download all type of files to run the project.

Example:
    $ python download-dataset.py
    ...
"""

import json
import os
import requests
import shutil

from tqdm import tqdm

with open(os.path.join(os.path.dirname(os.path.relpath(__file__)),
                       'content_reference.json')) as f:
    CONTENT = json.load(f)

ROOTPATH = os.path.join(os.path.dirname(os.path.relpath(__file__)),
                        'data')


def create_data_folder(folder: str):
    """create_data_folder

    Args:
        folder (str): folder to be created

    Raises:
        SystemExit: If user opts to not delete folder
    """
    path = os.path.join(ROOTPATH, folder)

    if os.path.exists(path):
        if get_input(f'{path} ALREADY EXIST!\nDo you want to delete? '):
            print(f'deleting {path}... ', end='')
            shutil.rmtree(path)
        else:
            raise SystemExit('Canceled by user!')

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
    path = os.path.join(ROOTPATH, folder)

    # the filename get from request
    filename = get_filename(url)

    # fullpath for the file
    fullpath = os.path.join(path, filename)

    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename, leave=False)
    with open(fullpath, 'wb') as f:
        for data in req.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        RuntimeError(f"ERROR DURING DOWNLOAD {url}")


def download_files(dataset_type: str):
    """Download all files from a dataset type

    Args:
        dataset_type (str): The dataset type which has to download files
    """
    print(f"Starting {dataset_type} files download...\n")
    for url in tqdm(CONTENT[dataset_type], desc=f'Downloading {dataset_type} files'):
        download(url, dataset_type)


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

    # Create a dict to hold all choices made by the user
    choices = dict()

    # Get from user which data has to be downloaded
    for dataset_type in CONTENT.keys():
        choice = get_input(f'Do you want download {dataset_type} data? ')

        # save the choice of the user
        choices[dataset_type] = choice

        if choice:
            create_data_folder(dataset_type)

    # iterate for each choice
    for dataset_type, choice in choices.items():
        print(f'{dataset_type} dataset files... ', end='')
        if choice:
            download_files(dataset_type)
        else:
            print(f'Nothing to be done!')

    print('All done! =]')
