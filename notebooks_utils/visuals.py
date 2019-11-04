import numpy as np
import seaborn as sns
import pandas as pd

from plotly import graph_objects as go
from matplotlib import pyplot as plt

from itertools import accumulate

import cv2


def barplot(ax, title, labels, legends, *args):
    r = np.arange(len(labels))
    width = 0.8/len(args)

    palette = sns.diverging_palette(255, 133, l=60, n=len(legends), center="dark")

    # set titles
    ax.set_title(title)

    # set ticks
    ax.set_xticks(r)
    ax.set_xticklabels(labels)

    for i, arg in enumerate(args):
        # ax.bar((r-0.8) + width*i, arg, width, label=legends[i], color=palette[i])
        widths = np.arange(len(args)) * width

        ax.bar(r + (widths[i] - np.median(widths)), arg, width, label=legends[i], color=palette[i])

    ax.set_axisbelow(True)
    ax.grid(which='major', axis='y', fillstyle='bottom')

    ax.legend(loc='upper right')


def stacked_bar(ax, title, labels, legends, *args, active_legends=True):
    r = np.arange(len(labels))
    barWidth = 0.8

    palette = sns.diverging_palette(255, 133, l=60, n=len(legends), center="dark")

    # set titles
    ax.set_title(title)

    # set ticks
    ax.set_xticks(r)
    ax.set_xticklabels(labels)

    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{x*100:.0f}%" for x in yticks])

    # func to calc the bottom
    calc_bottom = lambda a, b: [i+j for i, j in zip(a, b)]

    # draw first bar
    ax.bar(r, args[0], width=barWidth, label=legends[0], color=palette[0])

    if active_legends:
        for j, v in enumerate(args[0]):
            if v > 3:
                ax.text(-0.17 + j*1, (v - 8),
                        f"{v:2.2f}%", color='white', fontweight='bold')

    # draw bars after the first
    for i, bottom in enumerate(accumulate(args[:-1], calc_bottom), 1):
        ax.bar(r, args[i], bottom=bottom, width=barWidth, label=legends[i],
               color=palette[i])

        if active_legends:
            for j, v in enumerate(args[i]):
                if v > 9:
                    ax.text(-0.17 + j*1, (v - 8) + bottom[j],
                            f"{v:2.2f}%", color='white', fontweight='bold')

    # legend
    ax.legend(loc='lower right')


def sankey(df, height, width, classes_ref, title, pad=5, pos_leg=None):
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


def show_imgs(imgs, ax_array, df_imgs):
    for i, ax_i in enumerate(ax_array):
        for j, ax in enumerate(ax_i):
            img = imgs.iloc[i*2 + j]
            ref = df_imgs.loc[img.ImageID]

            # cleaning axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid()

            ax.set_title(f"{img.LabelSemantic.upper()}")

            img_path = ref.Path
            raw = cv2.imread(img_path)
            ax.imshow(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))


def show_bbox(imgs, ax_array, df_imgs, df_meta, print_others=True):
    for i, ax_i in enumerate(ax_array):
        for j, ax in enumerate(ax_i):
            meta = imgs.iloc[i*2 + j]
            ref = df_imgs.loc[meta.ImageID]

            # cleaning axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid()

            ax.set_title(f"{meta.LabelSemantic.upper()} - {'/'.join(ref.Path.split('/')[2:])}")

            img_path = ref.Path
            raw = cv2.imread(img_path)
            res = cv2.resize(raw, (int(raw.shape[1]/3), int(raw.shape[0]/3)))

            if print_others:
                for x in df_meta[df_meta.ImageID == meta.ImageID].itertuples():
                    add_bbox(res, x)

            add_bbox(res, meta, True)

            ax.imshow(cv2.cvtColor(res, cv2.COLOR_RGB2BGR))


def add_bbox(img, meta, principal=False):
    left_bottom = (int(img.shape[1] * meta.XMin),
                   int(img.shape[0] * meta.YMin))
    right_top = (int(img.shape[1] * meta.XMax),
                 int(img.shape[0] * meta.YMax))

    color = (55, 255, 0) if principal else (255, 210, 0)

    cv2.rectangle(img, left_bottom, right_top, color, 2)


def plot_heatmap_corr(df, principal, secondary, figsize):
    df_aux = _create_df_percentage(df, principal, secondary)

    fig, axes = plt.subplots(1, df_aux.shape[1], sharey=True, figsize=figsize)

    for i, col in enumerate(df_aux):
        sns.heatmap(df_aux[[col]], annot=True, ax=axes[i],
                    vmin=0, vmax=1, cmap="YlGnBu")
        axes[i].set_ylim(0, df_aux.shape[0])

    fig.text(0.5, 0, principal, ha='center', fontsize=14)
    fig.text(0.05, 0.5, secondary, va='center', rotation='vertical', fontsize=14)

    return fig, axes


def _create_df_percentage(df, principal, secondary):
    arr = []
    principal_values = df[principal].unique()
    secondary_values = df[secondary].unique()

    for line in principal_values:
        total = df[df[principal] == line].shape[0]
        for col in secondary_values:
            arr.append(df[(df[principal] == line)
                          & (df[secondary] == col)].shape[0] / total)

    arr_aux = np.transpose(np.reshape(arr, (len(principal_values),
                                            len(secondary_values))))

    return (pd.DataFrame(arr_aux, columns=principal_values, index=secondary_values)
              .sort_index(ascending=True))
