"""
This module is responsible to download all type of files to run the project.

Example:
    $ python download-dataset.py
    ...
"""

import os
import shutil


def create_data_folder(folder: str):
    """create_data_folder

    Args:
        folder (str): folder to be created

    Raises:
        SystemExit: If user opts to not delete folder
    """
    root = os.path.dirname(os.path.relpath(__file__))
    path = os.path.join(root, 'data', folder)

    if os.path.exists(path):
        if get_input(f'{path} ALREADY EXIST!\nDo you want to delete? '):
            print(f'deleting {path}...')
            shutil.rmtree(path)
        else:
            raise SystemExit('Exit by user!')

    print(f'creating {path}...\n')
    os.mkdir(path)


def download_datasets():
    """Download all datasets"""
    print("Starting datasets downloads...\n")


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
