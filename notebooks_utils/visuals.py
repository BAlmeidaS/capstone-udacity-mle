import numpy as np
import seaborn as sns

from plotly import graph_objects as go

from itertools import accumulate


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
