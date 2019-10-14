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


def extract(url: str, dataset_type: str):
    """Function that extract some tar file and after do it, delete it.

    Args:
        url (str): the url for download file
        folder (str): the relative path of folder where to write the file downloaded
    """
    # the path for save file - using data folder
    path = os.path.join(ROOTPATH, folder)

    # the filename get from request
    filename = get_filename(url)

    # fullpath for the file
    fullpath = os.path.join(path, filename)

    # open the tar file
    tar = tarfile.open(fullpath)

    # extract all files
    tar.extractall()

    # close file
    tar.close()

    # remove tar file
    os.remove(fullpath)


def download_files(dataset_type: str, urls: list):
    """Download all files from a dataset type

    Args:
        dataset_type (str): The dataset type which has to download files
        urls (list): list of urls to download files
    """
    print(f"Starting {dataset_type} files download...\n")
    for url in tqdm(urls, desc=f'Downloading {dataset_type} files'):
        download(url, dataset_type)


def download_extract_files(dataset_type: str, urls: list):
    """Download all files from a dataset type and extract one by one.
    Removing all tar files after end each extraction

    Args:
        dataset_type (str): The dataset type which has to download files
        urls (list): list of urls to download files
    """
    print(f"Starting {dataset_type} files download...\n")
    for url in tqdm(urls, desc=f'Downloading and extracting {dataset_type} files'):
        download(url, dataset_type)
        extract(url, dataset_type)


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
    for set_type, content in CONTENT.items():
        if get_input(f"Do you want download {set_type} data?\n({content['info']}): "):
            # save the choice of the user
            choices[set_type] = content["urls"]

            create_data_folder(set_type)

    # iterate for each choice
    for set_type, urls in choices.items():
        print(f'{set_type} data files... ', end='')
        if set_type == 'METADATA':
            download_files(set_type, urls)
        else:
            download_extract_files(set_type, urls)


    print('All done! =]')
