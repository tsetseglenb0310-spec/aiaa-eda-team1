# aiaa-eda-team1

AI Academy Asia, **1-р баг** — Монголын нээлттэй өгөгдлийн EDA хийх туслах функцууд.

**PyPI:** https://pypi.org/project/aiaa-eda-team1/ — хувилбар 0.1.0, 2026.10.23-нд байршуулсан. Суулгах: `pip install aiaa-eda-team1`.

Төсөл: «Монгол Улсын хүн амын шилжилт хөдөлгөөн: аймгуудын харьцуулалт» (ҮСХ, 1212.mn, хүснэгт DT_NSO_0300_040V1).

> Багийн дугаар өөр бол `aiaa-eda-team1` → `aiaa-eda-team<N>` (pyproject.toml, хавтасны нэр `aiaa_eda_team<N>`, notebook дахь `import`) гэж солино.

## Суулгах

```bash
# Colab
!pip install -q aiaa-eda-team1

# PyPI-д байршаагүй үед GitHub-аас
!pip install -q git+https://github.com/tsetseglenb0310-spec/aiaa-eda-team1.git

# Хөгжүүлэх (VS Code, .venv идэвхтэй)
pip install -e .
```

```python
import aiaa_eda_team1 as et
et.missing_report(df)
```

## Функцууд

### cleaning.py — цэвэрлэгээ (Гишүүн 1)

| Функц | Параметр | Буцаах утга |
|---|---|---|
| `missing_report(df)` | `df` | DataFrame[column, dtype, missing, missing_pct] |
| `iqr_outliers(df, col, k=1.5)` | хүснэгт, тоон багана, үржүүлэгч | dict {low, high, n_outliers, rows} |
| `strip_text_columns(df)` | хүснэгт | (df, өөрчлөгдсөн нүдний тоо) |
| `drop_duplicate_rows(df)` | хүснэгт | (df, устгасан мөрийн тоо) |
| `split_totals(df, name_col, total_names)` | хүснэгт, нэрийн багана, дүнгийн нэрс | (territories, totals) |
| `flows_wide_to_long(df, indicator_col, name_col, in_label, out_label)` | өргөн хүснэгт, баганын нэрс, үзүүлэлтийн нэрс | DataFrame[aimag, year, in_mig, out_mig] |
| `mark_both_zero_as_missing(df, cols)` | хүснэгт, баганууд | (df, тэмдэглэсэн мөрийн тоо, мөрүүд) |
| `add_region(df, name_col="aimag", mapping=None)` | хүснэгт | df + `region` багана |
| `REGION_OF` | — | {аймаг: бүс} толь |

### stats.py — тоон шинжилгээ (Гишүүн 1)

| Функц | Параметр | Буцаах утга |
|---|---|---|
| `basic_stats(x, ddof=1)` | массив / Series | dict {n, mean, median, std, min, q1, q3, iqr, max} |
| `group_rate(df, group_col, target_col)` | хүснэгт, бүлэг, 0/1 багана | DataFrame[group, n, rate] |
| `rank_table(df, value_col, name_col="aimag", ascending=False, extra_cols=None)` | нэг оны хүснэгт | DataFrame[rank, name, …, value] |
| `rank_gap(df, col_a, col_b, name_col="aimag")` | хоёр багана | DataFrame[name, rank_a, rank_b, gap] |
| `national_balance(totals, indicator_col, name_col, total_name, in_label, out_label)` | дүнгийн мөрүүд | DataFrame[year, in_total, out_total, diff] |
| `period_summary(df, group_col, value_col, since=None)` | урт хүснэгт | DataFrame[group, mean, min, max, years_positive, years_total] |
| `consistency_checks(long_df, totals, …, total_names, region_of, n_expected=None)` | урт ба дүнгийн хүснэгт | (dict{шалгалт: bool}, details) |
| `fmt(v, sign=False)` | тоо | '32 030' / '+19 034' |

### plots.py — график (Гишүүн 2). `path="figures/01_….png"` өгвөл файлд хадгална, Axes буцаана.

| Функц | Юу зурна |
|---|---|
| `plot_hist(df, col, bins=10, path, title, xlabel, source)` | гистограмм + медиан, дундажийн шугам |
| `plot_box(df, col, by=None, path, title, xlabel, source, label_col, label_rows)` | boxplot + цэгүүд, outlier-ийн нэр |
| `plot_corr(df, cols=None, path, title, source)` | корреляцийн heatmap |
| `plot_bar(df, name_col, value_col, path, title, xlabel, source, color_by_sign=False, top=None)` | хэвтээ эрэмбийн bar chart, тэг шугамтай |
| `plot_grouped_bar(df, name_col, cols, labels, path, title, ylabel, source)` | хосолсон баганат график |
| `plot_line(df, x, y, group=None, labels=None, path, title, ylabel, source, vline, vline_label, annotate_max, fill_between)` | оноор line chart |

### extra.py — нэмэлт (Гишүүн 2)

| Функц | Юу зурна |
|---|---|
| `plot_scatter(df, x, y, label_col=None, path, title, source)` | тархалтын график |
| `plot_small_multiples(df, x, y, group, order=None, path, title, ylabel, source)` | бүлэг бүрт тусдаа панел, өөрийн масштабтай |

## Хөгжүүлэх, байршуулах (гарын авлагын дагуу)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
python -m pip install pandas numpy matplotlib seaborn build twine
pip install -e .
python -c "import aiaa_eda_team1 as et; print(et.basic_stats([1, 2, 3]))"

rmdir /s /q dist
python -m build
python -m twine check dist/*
python -m twine upload dist/*     # username: __token__, password: pypi-… token
```

Шинэ функц нэмсэн бол `__init__.py`-д нэрийг нь нэмж, `pyproject.toml`-д хувилбарыг өсгөнө (0.1.0 → 0.1.1 алдаа засвар, 0.2.0 шинэ функц).

## Git дүрэм

- Гишүүн бүр өөрийн branch: `git checkout -b <нэр>`; ажил бүрийн өмнө `git pull origin main`.
- Зөвхөн өөрийн файлаа add хийнэ: `git add aiaa_eda_team1/stats.py` (`git add .` хэрэглэхгүй).
- Pull Request нээнэ, ахлагч main руу merge хийнэ.

| Гишүүн | Файл |
|---|---|
| 1 | `cleaning.py`, `stats.py` |
| 2 (сангийн ахлагч) | `plots.py`, `extra.py`, `__init__.py`, `pyproject.toml`, PyPI |
| 3 | шинжилгээний repo: `analysis.ipynb`, `figures/` |
| 4 (багийн ахлагч) | шинжилгээний repo: `README.md`, танилцуулга |
