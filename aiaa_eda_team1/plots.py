"""plots.py — графикийн функцууд (Гишүүн 2). Matplotlib + Seaborn.

Бүх функц path өгвөл PNG файлд хадгална (dpi=150), үргэлж Axes буцаана.
Он, нэгж, эх сурвалжийг title / xlabel / source параметрээр заана.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BLUE, RED, GREY, DARK, TEAL = "#1f5fa8", "#c8423c", "#8a8a8a", "#222222", "#2a9d8f"
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def _thousands(v, _=None):
    return f"{v:,.0f}".replace(",", " ")


def _finish(fig, ax, title, xlabel, ylabel, source, path):
    """Дотоод туслах: гарчиг, тэнхлэг, эх сурвалж, хадгалах."""
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if source:
        fig.text(0.01, 0.005, source, fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.03 if source else 0, 1, 1))
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return ax


def plot_hist(df, col, bins=10, path=None, title=None, xlabel=None, source=None, show_center=True):
    """Нэг тоон баганын гистограмм; медиан ба дундажийн шугамтай.

    Параметр: df, col — багана; bins — интервалын тоо; path — хадгалах зам; title, xlabel, source — бичээс.
    Буцаана: Axes.
    """
    x = df[col].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.histplot(x, bins=bins, color=BLUE, edgecolor="white", ax=ax)
    if show_center:
        ax.axvline(np.median(x), color=DARK, linestyle="--", linewidth=1.5, label=f"медиан {_thousands(np.median(x))}")
        ax.axvline(np.mean(x), color=RED, linewidth=1.5, label=f"дундаж {_thousands(np.mean(x))}")
        ax.legend(frameon=False)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    return _finish(fig, ax, title, xlabel if xlabel is not None else col, "Тоо", source, path)


def plot_box(df, col, by=None, path=None, title=None, xlabel=None, source=None, label_col=None, label_rows=None):
    """Boxplot (хайрцган график) + цэгүүд. by өгвөл бүлгээр.

    Параметр: df, col — багана; by — бүлэглэх багана (сонголттой); label_col/label_rows — нэрээр тэмдэглэх мөрүүд.
    Буцаана: Axes.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if by is None:
        sns.boxplot(x=df[col], color="#d9e4f2", width=0.4, fliersize=0, ax=ax)
        sns.stripplot(x=df[col], color=BLUE, size=6, alpha=0.8, jitter=0.15, ax=ax)
        if label_col is not None and label_rows is not None:
            for i, (_, row) in enumerate(label_rows.iterrows()):
                ax.annotate(row[label_col], (row[col], 0), xytext=(-6, 16) if i % 2 == 0 else (0, -70),
                            textcoords="offset points", ha="right" if i % 2 == 0 else "center", fontsize=9, color=DARK,
                            arrowprops=dict(arrowstyle="-", color=GREY))
    else:
        sns.boxplot(data=df, x=col, y=by, color="#d9e4f2", fliersize=0, ax=ax)
        sns.stripplot(data=df, x=col, y=by, color=BLUE, size=5, alpha=0.8, ax=ax)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    return _finish(fig, ax, title, xlabel if xlabel is not None else col, "" if by is None else by, source, path)


def plot_corr(df, cols=None, path=None, title=None, source=None):
    """Тоон багануудын корреляцийн дулааны зураг (heatmap).

    Параметр: df; cols — баганууд (анхдагч: бүх тоон багана); path, title, source.
    Буцаана: Axes.
    """
    data = df[cols] if cols else df.select_dtypes("number")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
    return _finish(fig, ax, title, "", "", source, path)


def plot_bar(df, name_col, value_col, path=None, title=None, xlabel=None, source=None,
             color_by_sign=False, color=BLUE, top=None):
    """Хэвтээ эрэмбийн bar chart: хамгийн их утга дээр, утгын бичээстэй, тэг шугамтай.

    Параметр: df — нэг оны хүснэгт; name_col, value_col; color_by_sign — эерэг цэнхэр / сөрөг улаан;
              top — зөвхөн эхний N мөр; path, title, xlabel, source.
    Буцаана: Axes.
    """
    data = df.sort_values(value_col, ascending=False)
    if top:
        data = data.head(top)
    data = data.sort_values(value_col, ascending=True)
    values = data[value_col].to_numpy(dtype=float)
    colors = [BLUE if v > 0 else RED for v in values] if color_by_sign else [color] * len(values)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(data[name_col], values, color=colors)
    span = values.max() - min(values.min(), 0)
    for i, v in enumerate(values):
        ax.text(v + span * 0.01 if v >= 0 else v - span * 0.01, i, _thousands(v),
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=DARK)
    ax.axvline(0, color=DARK, linewidth=0.8)
    lo = min(values.min(), 0) - span * 0.12 if values.min() < 0 else 0
    ax.set_xlim(lo, values.max() + span * 0.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    return _finish(fig, ax, title, xlabel if xlabel is not None else value_col, "", source, path)


def plot_grouped_bar(df, name_col, cols, labels=None, path=None, title=None, ylabel="Хүн", source=None):
    """Хосолсон баганат график: нутаг бүрт хэд хэдэн багана зэрэгцүүлэн (жишээ: ирсэн ба явсан).

    Параметр: df; name_col — нэрийн багана; cols — утгын багануудын жагсаалт; labels — легендийн нэрс.
    Буцаана: Axes.
    """
    labels = labels or cols
    n, k = len(df), len(cols)
    width = 0.8 / k
    x = np.arange(n)
    palette = [BLUE, RED, TEAL, "#d9a441"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (c, lab) in enumerate(zip(cols, labels)):
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, df[c], width, color=palette[j % len(palette)], label=lab)
        for xi, v in zip(pos, df[c]):
            ax.text(xi, v + df[cols].to_numpy().max() * 0.01, _thousands(v), ha="center", fontsize=9, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(df[name_col], fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    ax.legend(frameon=False)
    return _finish(fig, ax, title, "", ylabel, source, path)


def plot_line(df, x, y, group=None, labels=None, path=None, title=None, ylabel=None, source=None,
              vline=None, vline_label=None, annotate_max=False, fill_between=None):
    """Оноор line chart. group өгвөл бүлэг бүрт нэг шугам; y жагсаалт бол нэг объектын хэд хэдэн шугам.

    Параметр: df — урт хүснэгт; x — он; y — багана эсвэл багануудын жагсаалт; group — бүлэглэх багана; labels — легендийн нэрс;
              vline — босоо тасархай шугамын x утга; annotate_max — y-ийн максимумыг бичээслэх;
              fill_between — (y1, y2) хоёр шугамын хоорондыг будах.
    Буцаана: Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = [BLUE, RED, TEAL, "#d9a441", GREY]
    ys = y if isinstance(y, (list, tuple)) else [y]
    labels = list(labels) if labels else ys
    if group is None:
        d = df.sort_values(x)
        for j, col in enumerate(ys):
            ax.plot(d[x], d[col], color=palette[j % len(palette)], linewidth=2.2, marker="o", markersize=3, label=labels[j])
        if fill_between:
            a, b = fill_between
            ax.fill_between(d[x], d[a], d[b], where=d[a] >= d[b], color=BLUE, alpha=0.08, interpolate=True)
        if annotate_max:
            col = ys[0]
            i = d[col].idxmax()
            ax.annotate(f"{int(d.loc[i, x])}: {_thousands(d.loc[i, col])}", xy=(d.loc[i, x], d.loc[i, col]),
                        xytext=(d.loc[i, x] - 7, d.loc[i, col] * 1.02), fontsize=9, color=DARK,
                        arrowprops=dict(arrowstyle="-", color=GREY))
    else:
        for j, (name, part) in enumerate(df.groupby(group)):
            part = part.sort_values(x)
            ax.plot(part[x], part[ys[0]], color=palette[j % len(palette)], linewidth=2.2, marker="o", markersize=3, label=str(name))
    ax.axhline(0, color=DARK, linewidth=0.8)
    if vline is not None:
        ax.axvline(vline, color=GREY, linestyle=":", linewidth=1)
        if vline_label:
            ax.text(vline + 0.3, ax.get_ylim()[1] * 0.5, vline_label, fontsize=8, color=GREY, va="top")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands))
    ax.legend(frameon=False, loc="upper left")
    return _finish(fig, ax, title, "Он", ylabel if ylabel is not None else ys[0], source, path)
