"""extra.py — нэмэлт графикийн функцууд (Гишүүн 2 / ахлагч).
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from .plots import BLUE, RED, GREY, DARK, TEAL, _thousands, _finish


def plot_scatter(df, x, y, label_col=None, path=None, title=None, source=None):
    """Хоёр тоон баганын тархалтын график (scatter); label_col өгвөл цэг бүрийг нэрлэнэ.

    Параметр: df; x, y — баганууд; label_col — нэрийн багана; path, title, source.
    Буцаана: Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x], df[y], s=60, color=BLUE, edgecolor="white")
    if label_col:
        for _, r in df.iterrows():
            ax.text(r[x], r[y], str(r[label_col]), fontsize=8, color=DARK)
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    return _finish(fig, ax, title, x, y, source, path)


def plot_small_multiples(df, x, y, group, order=None, path=None, title=None, ylabel=None, source=None):
    """Бүлэг бүрт тусдаа панел (өөрийн масштабтай), нэг багана дараалсан line chart.

    Параметр: df — урт хүснэгт; x — он; y — утга; group — бүлэг; order — панелийн дараалал (нэрсийн жагсаалт).
    Буцаана: массив Axes.
    """
    names = order or sorted(df[group].unique())
    colors = [BLUE, TEAL, RED, "#d9a441", GREY]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), sharex=True)
    for ax, name, color in zip(axes, names, colors):
        part = df[df[group] == name].sort_values(x)
        ax.plot(part[x], part[y], color=color, linewidth=2.2, marker="o", markersize=3)
        ax.fill_between(part[x], part[y], 0, color=color, alpha=0.12)
        ax.axhline(0, color=DARK, linewidth=0.8)
        ax.set_title(str(name), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel or y)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    axes[-1].set_xlabel("Он")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    if source:
        fig.text(0.01, 0.005, source, fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return axes
