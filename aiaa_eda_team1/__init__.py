"""aiaa-eda-team1 — Монголын нээлттэй өгөгдлийн EDA хийх туслах функцууд.

Хэрэглээ (Colab):
    !pip install -q aiaa-eda-team1
    import aiaa_eda_team1 as et
    et.missing_report(df)
"""

from .cleaning import (
    missing_report,
    iqr_outliers,
    strip_text_columns,
    drop_duplicate_rows,
    split_totals,
    flows_wide_to_long,
    mark_both_zero_as_missing,
    add_region,
    REGION_OF,
)
from .stats import (
    basic_stats,
    group_rate,
    rank_table,
    rank_gap,
    national_balance,
    period_summary,
    consistency_checks,
    fmt,
)
from .plots import (
    plot_hist,
    plot_box,
    plot_corr,
    plot_bar,
    plot_grouped_bar,
    plot_line,
)
from .extra import plot_scatter, plot_small_multiples

__version__ = "0.1.0"
