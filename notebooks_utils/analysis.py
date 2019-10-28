import os
import csv

import pandas as pd
import numpy as np

from plotly import graph_objects as go

from project.download_content import DATAPATH


classes_csv_path = os.path.join(DATAPATH,
                                'METADATA',
                                'class-descriptions-boxable.csv')

with open(classes_csv_path, mode='r') as f:
    classes_map = {rows[0]: rows[1] for rows in csv.reader(f)}


def semantic_name(encoded_name: str):
    return classes_map[encoded_name]


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


def generate_df(tree: dict, some_class: str, *args) -> list:
    paths = []

    def find_paths_recursive(tree, *args):
        if tree['LabelName'] == some_class:
            paths.append((*args, some_class))
        elif 'Subcategory' in tree.keys():
            for subcat in tree['Subcategory']:
                find_paths_recursive(subcat, *args, tree['LabelName'])

    find_paths_recursive(tree)

    return paths


def images_downloaded(data_type: str) -> pd.DataFrame:
    temp_array = []
    path = os.path.join(DATAPATH, data_type)
    for root, dirs, files in os.walk(path):
        temp_array += files

    df = pd.DataFrame(temp_array, columns=['ImageID'])
    df.ImageID = df.ImageID.apply(lambda x: x[:-4])

    return df


def all_images_downloaded() -> bool:
    """return true if all folder of all images are created, otherwise False"""
    return all([os.path.exists(os.path.join(DATAPATH, data_type))
                for data_type in ['TRAIN', 'VALIDATION', 'TEST']])


def check_images_download():
    """raise an exception if images were not downloaded"""
    if not all_images_downloaded():
        raise RuntimeError('You did not download images, this cell will not run properly!')


def create_sankey(df, height, width, classes_ref, title, pad=5, pos_leg=None):
    fig = go.Figure(data=[go.Sankey(
        name='bruno',
        valueformat="",
        valuesuffix="",
        node=dict(
            groups=[[1]]*(classes_ref.Label.nunique() - 1),
            pad=pad,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=classes_ref.sort_values(by=['Id']).Label.unique(),
            color=(classes_ref.sort_values(by=['Id'])
                              .drop_duplicates(['Id'])
                              .Leaf
                              .apply(lambda x: "#c8d419" if x else "#f63a76"))
        ),
        link=dict(
            source=df.IdParent.values,
            target=df.Id.values,
            value=np.ones(df.shape[0]),
            color="#ebebeb"))])

    fig.update_layout(title_text=title, font_size=10)

    fig.update_layout(height=height, width=width)

    fig.update_layout(showlegend=True)

    if pos_leg:
        fig.update_layout(go.Layout(
            annotations=[
                go.layout.Annotation(
                    text='<b>Leaf</b>',
                    align='left',
                    showarrow=False,
                    x=pos_leg[0][0],
                    y=pos_leg[0][1],
                    font=dict(
                        size=12,
                        color='#444444'
                    ),
                    bordercolor='#000000',
                    bgcolor='#c8d419',
                    borderwidth=1
                ),
                go.layout.Annotation(
                    text='<b>Path</b>',
                    align='left',
                    showarrow=False,
                    font=dict(
                        size=12,
                        color='#ffffff'
                    ),
                    x=pos_leg[1][0],
                    y=pos_leg[1][1],
                    bordercolor='#000000',
                    bgcolor='#f63a76',
                    borderwidth=1)]))

    return fig


def tabularize_hierarchy_dict(graph: dict, classes) -> list:
    arr = []

    def flatten_dict(subgraph: dict,
                     parent_id: int = -1,
                     parent_label: str = None,
                     depth: int = 0):

        node_label = subgraph['LabelName']
        node_id = classes[classes.class_name == node_label].index[0]
        leaf = True

        if 'Subcategory' in subgraph.keys():
            leaf = False
            for subcat in subgraph['Subcategory']:
                flatten_dict(subcat,
                             node_id,
                             node_label,
                             depth+1)

        arr.append([node_id, node_label, parent_id, parent_label, depth, leaf])

    flatten_dict(graph)

    return arr
