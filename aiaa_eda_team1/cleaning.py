"""cleaning.py — өгөгдөл шалгах, цэвэрлэх функцууд (Гишүүн 1).

Бүх функц анхны DataFrame-ийг өөрчлөхгүй, хуулбар буцаана.
"""

import numpy as np
import pandas as pd

# Аймаг → бүс (ҮСХ-ны 1212.mn хүснэгтийн бүсчлэл). Улаанбаатар бүсэд ордоггүй.
REGION_OF = {
    "Баян-Өлгий": "Баруун бүс", "Говь-Алтай": "Баруун бүс", "Завхан": "Баруун бүс",
    "Увс": "Баруун бүс", "Ховд": "Баруун бүс",
    "Архангай": "Хангайн бүс", "Баянхонгор": "Хангайн бүс", "Булган": "Хангайн бүс",
    "Орхон": "Хангайн бүс", "Өвөрхангай": "Хангайн бүс", "Хөвсгөл": "Хангайн бүс",
    "Говьсүмбэр": "Төвийн бүс", "Дархан-Уул": "Төвийн бүс", "Дорноговь": "Төвийн бүс",
    "Дундговь": "Төвийн бүс", "Өмнөговь": "Төвийн бүс", "Сэлэнгэ": "Төвийн бүс", "Төв": "Төвийн бүс",
    "Дорнод": "Зүүн бүс", "Сүхбаатар": "Зүүн бүс", "Хэнтий": "Зүүн бүс",
    "Улаанбаатар": "Улаанбаатар",
}


def missing_report(df):
    """Багана бүрийн дутуу утгын тайлан.

    Параметр: df — DataFrame.
    Буцаана: DataFrame[column, dtype, missing, missing_pct] — дутуу утга ихтэйгээс нь эрэмбэлсэн.
    """
    rep = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isnull().sum().values,
    })
    rep["missing_pct"] = (rep["missing"] / len(df) * 100).round(1)
    return rep.sort_values("missing", ascending=False).reset_index(drop=True)


def iqr_outliers(df, col, k=1.5):
    """1.5·IQR дүрмээр outlier (хэт утга) олох.

    Параметр: df — DataFrame; col — тоон баганын нэр; k — IQR-ийн үржүүлэгч (анхдагч 1.5).
    Буцаана: dict {"low": доод хил, "high": дээд хил, "n_outliers": тоо, "rows": outlier мөрүүд (DataFrame)}.
    """
    x = df[col].dropna().to_numpy(dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    rows = df[(df[col] < low) | (df[col] > high)].copy()
    return {"low": float(low), "high": float(high), "n_outliers": int(len(rows)), "rows": rows}


def strip_text_columns(df):
    """Текст багануудын захын зайг арилгах.

    Параметр: df — DataFrame.
    Буцаана: (df_цэвэр, өөрчлөгдсөн нүдний тоо).
    """
    out = df.copy()
    changed = 0
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype) in ("string", "str"):
            stripped = out[c].astype(str).str.strip()
            changed += int((stripped != out[c].astype(str)).sum())
            out[c] = stripped
    return out, changed


def drop_duplicate_rows(df):
    """Бүрэн давхардсан мөрийг устгах.

    Параметр: df — DataFrame.
    Буцаана: (df_давхардалгүй, устгасан мөрийн тоо).
    """
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def split_totals(df, name_col, total_names):
    """Дүнгийн мөрүүдийг (улс, бүс) нутгуудаас тусгаарлах.

    Параметр: df — DataFrame; name_col — нутгийн нэрийн багана; total_names — дүнгийн мөрийн нэрсийн жагсаалт.
    Буцаана: (territories, totals) — хоёр DataFrame.
    """
    is_total = df[name_col].isin(total_names)
    return df[~is_total].copy(), df[is_total].copy()


def flows_wide_to_long(df, indicator_col, name_col, in_label, out_label, name_out="aimag"):
    """Өргөн хүснэгтийг (он баганаар) урт хүснэгт болгох: нэг мөр = нэг нутгийн нэг он.

    Параметр: df — territories (нутгийн мөрүүд); indicator_col — үзүүлэлтийн багана; name_col — нутгийн багана;
              in_label / out_label — ирсэн ба явсан үзүүлэлтийн нэр; name_out — гаралтын нутгийн баганын нэр.
    Буцаана: DataFrame[aimag, year, in_mig, out_mig], (нутаг, он)-оор эрэмбэлсэн.
    Арга: он бүрт нэг жижиг хүснэгт үүсгэж (dict + map) pd.concat-аар нэгтгэнэ; merge/melt ашиглахгүй.
    """
    year_cols = [c for c in df.columns if str(c).isdigit()]
    in_rows = df[df[indicator_col] == in_label]
    out_rows = df[df[indicator_col] == out_label]
    tables = []
    for y in year_cols:
        out_by_name = dict(zip(out_rows[name_col], out_rows[y]))
        t = pd.DataFrame({name_out: in_rows[name_col].values, "year": int(y), "in_mig": in_rows[y].values})
        t["out_mig"] = t[name_out].map(out_by_name)
        tables.append(t)
    long = pd.concat(tables, ignore_index=True)
    return long.sort_values([name_out, "year"]).reset_index(drop=True)


def mark_both_zero_as_missing(df, cols):
    """Заасан бүх багана зэрэг тэг байгаа мөрүүдэд NaN (өгөгдөлгүй) тавих.

    Параметр: df — DataFrame; cols — баганын нэрсийн жагсаалт (жишээ ["in_mig", "out_mig"]).
    Буцаана: (df_шинэ, тэмдэглэсэн мөрийн тоо, тэмдэглэсэн мөрүүд).
    Үндэслэл: бүх урсгал зэрэг тэг байх нь ихэвчлэн бүртгэлгүй байсныг заана, тэг шилжилтийг биш.
    """
    out = df.copy()
    mask = np.ones(len(out), dtype=bool)
    for c in cols:
        mask &= (out[c] == 0).to_numpy()
    marked = out[mask].copy()
    out.loc[mask, cols] = np.nan
    return out, int(mask.sum()), marked


def add_region(df, name_col="aimag", mapping=None):
    """Нутгийн нэрээр бүсийн багана нэмэх.

    Параметр: df — DataFrame; name_col — нутгийн нэрийн багана; mapping — {нутаг: бүс} толь (анхдагч REGION_OF).
    Буцаана: df-ийн хуулбар, "region" баганатай.
    """
    out = df.copy()
    out["region"] = out[name_col].map(mapping or REGION_OF)
    return out
