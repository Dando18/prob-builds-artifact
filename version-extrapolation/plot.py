""" Plotting utilities.
"""
# std imports
from typing import Iterable, Optional

# tpl imports
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    figsize: tuple = (12, 8),
    output: Optional[str] = None,
    **kwargs
):
    """ Plot a line chart """
    plt.clf()
    sns.set()
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df, x=x, y=y, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_title:
        ax.get_legend().set_title(legend_title)
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    label: bool = False,
    xtick_rotation: int = 0,
    figsize: tuple = (12, 8),
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    output: Optional[str] = None,
    **kwargs
):
    """ Plot a bar chart """
    plt.close()
    plt.clf()
    sns.set()
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x=x, y=y, **kwargs)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    if legend_title:
        ax.get_legend().set_title(legend_title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if label:
        ax.bar_label(ax.containers[1], fmt='%.3g', label_type='center')
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()



def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    label: bool = False,
    label_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    figsize: tuple = (12, 8),
    scatter_labels: bool = False,
    output: Optional[str] = None,
    **kwargs
):
    """ Plot a scatter plot """
    sns.set()
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_title:
        ax.get_legend().set_title(legend_title)

    ax.set_xlim(0, 0.024)
    ax.set_ylim(0, 0.024)

    if label and label_col:
        labels = []
        for line in range(0, df.shape[0]):
            if line % 4 != 0:
                continue
            labels.append(ax.text(
                df[x][line] + 0.0001,
                df[y][line],
                df[label_col][line],
                horizontalalignment="left",
                size="small",
                color="black"
            ))
        
        if scatter_labels and labels:
            from adjustText import adjust_text
            adjust_text(labels)

    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_histogram(
    df: pd.DataFrame,
    x: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 8),
    output: Optional[str] = None,
    **kwargs
):
    """ Plot a histogram """
    sns.set()
    plt.figure(figsize=figsize)
    ax = sns.histplot(data=df, x=x, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if output:
        plt.savefig(output)
    else:
        plt.show()