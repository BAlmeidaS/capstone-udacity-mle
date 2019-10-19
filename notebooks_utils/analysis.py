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
    return classes_map[encoded_name]


def flatten(coll: object):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def count_recursive(tree: dict) -> int:
    nodes = []

    def recursion(tree, count: int = 0):
        nodes.append(tree['LabelName'])

        if 'Subcategory' in tree.keys():
            for subcat in tree['Subcategory']:
                count = recursion(subcat, count)
        return count + 1

    return recursion(tree), nodes


def find_path_to_node(tree: dict, some_class: str) -> list:
    if tree['LabelName'] == some_class:
        return [some_class]
    elif 'Subcategory' in tree.keys():
        paths = []
        for subcat in tree['Subcategory']:
            path = find_path_to_node(subcat, some_class)
            if path:
                full_path = list(flatten([tree['LabelName']] + path))
                paths.append(full_path)
        return paths
