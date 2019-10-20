import os
import csv
from collections.abc import Iterable

from project.download_content import DATAPATH

classes_csv_path = os.path.join(DATAPATH,
                                'METADATA',
                                'class-descriptions-boxable.csv')

with open(classes_csv_path, mode='r') as f:
    classes_map = {rows[0]: rows[1] for rows in csv.reader(f)}


def semantic_name(encoded_name: str):
    try:
        name = classes_map[encoded_name]
    except KeyError:
        name = 'Entity'
    return name


def count_recursive(tree: dict) -> int:
    nodes = []

    def recursion(tree, count: int = 0):
        nodes.append(tree['LabelName'])

        if 'Subcategory' in tree.keys():
            for subcat in tree['Subcategory']:
                count = recursion(subcat, count)
        return count + 1

    return recursion(tree), nodes


def node_path(tree: dict, some_class: str, *args) -> list:
    paths = []

    def find_paths_recursive(tree, *args):
        if tree['LabelName'] == some_class:
            paths.append((*args, some_class))
        elif 'Subcategory' in tree.keys():
            for subcat in tree['Subcategory']:
                find_paths_recursive(subcat, *args, tree['LabelName'])

    find_paths_recursive(tree)

    return paths
