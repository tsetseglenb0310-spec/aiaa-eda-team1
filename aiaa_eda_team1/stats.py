"""stats.py — тоон шинжилгээ, эрэмбэ, шалгалтын функцууд (Гишүүн 1).
"""

import numpy as np
import pandas as pd


def fmt(v, sign=False):
    """Тоог мянгатын зайтай текст болгох: 32030 -> '32 030'; sign=True бол +/− тэмдэгтэй.

    Параметр: v — тоо; sign — тэмдэг харуулах эсэх.
    Буцаана: str.
    """
    s = f"{v:+,.0f}" if sign else f"{v:,.0f}"
    return s.replace(",", " ")


def basic_stats(x, ddof=1):
    """Тоон массивын үндсэн статистик (NumPy).

    Параметр: x — жагсаалт, Series эсвэл массив; ddof — std-ийн чөлөөт зэрэг (1 = түүврийн).
    Буцаана: dict {n, mean, median, std, min, q1, q3, iqr, max}.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(len(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=ddof)), "min": float(np.min(arr)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "max": float(np.max(arr)),
    }


def group_rate(df, group_col, target_col):
    """Бүлэг бүрт зорилтот баганын дундаж (0/1 бол эзлэх хувь).

    Параметр: df — DataFrame; group_col — бүлэглэх багана; target_col — 0/1 эсвэл тоон багана.
    Буцаана: DataFrame[group_col, n, rate] — rate буурахаар эрэмбэлсэн.
    """
    g = (df.groupby(group_col)
           .agg(n=(target_col, "count"), rate=(target_col, "mean"))
           .reset_index()
           .sort_values("rate", ascending=False)
           .reset_index(drop=True))
    g["rate"] = g["rate"].round(3)
    return g


def rank_table(df, value_col, name_col="aimag", ascending=False, extra_cols=None):
    """Нэг баганаар эрэмбэлсэн хүснэгт, 1-ээс эхэлсэн rank баганатай.

    Параметр: df — нэг оны хүснэгт; value_col — эрэмбэлэх багана; name_col — нэрийн багана;
              ascending — өсөхөөр эсэх; extra_cols — нэмж харуулах баганууд.
    Буцаана: DataFrame[rank, name_col, (extra_cols), value_col].
    """
    cols = [name_col] + list(extra_cols or []) + ([value_col] if value_col not in (extra_cols or []) else [])
    out = df.sort_values(value_col, ascending=ascending)[cols].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rank_gap(df, col_a, col_b, name_col="aimag"):
    """Хоёр баганаар эрэмбэлсэн байрын зөрүү (жишээ: ирсний байр vs цэвэр шилжилтийн байр).

    Параметр: df — нэг оны хүснэгт; col_a, col_b — харьцуулах баганууд; name_col — нэр.
    Буцаана: DataFrame[name_col, col_a, col_b, rank_a, rank_b, gap] — gap буурахаар эрэмбэлсэн (gap = rank_b − rank_a).
    """
    out = df[[name_col, col_a, col_b]].copy()
    out["rank_a"] = out[col_a].rank(ascending=False).astype(int)
    out["rank_b"] = out[col_b].rank(ascending=False).astype(int)
    out["gap"] = out["rank_b"] - out["rank_a"]
    return out.sort_values("gap", ascending=False).reset_index(drop=True)


def national_balance(totals, indicator_col, name_col, total_name, in_label, out_label):
    """Улсын дүнгийн мөрөөр он бүрийн ирсэн − явсан зөрүү (бүртгэлийн харьцуулах боломжийн шалгалт).

    Параметр: totals — дүнгийн мөрүүд; indicator_col, name_col — баганын нэрс; total_name — улсын дүнгийн нэр;
              in_label, out_label — үзүүлэлтийн нэрс.
    Буцаана: DataFrame[year, in_total, out_total, diff], оноор эрэмбэлсэн.
    """
    year_cols = [c for c in totals.columns if str(c).isdigit()]
    nat_in = totals[(totals[name_col] == total_name) & (totals[indicator_col] == in_label)][year_cols].iloc[0]
    nat_out = totals[(totals[name_col] == total_name) & (totals[indicator_col] == out_label)][year_cols].iloc[0]
    out = pd.DataFrame({"year": [int(y) for y in year_cols], "in_total": nat_in.values, "out_total": nat_out.values})
    out["diff"] = out["in_total"] - out["out_total"]
    return out.sort_values("year").reset_index(drop=True)


def period_summary(df, group_col, value_col, since=None):
    """Хугацааны хураангуй: бүлэг бүрт дундаж, мин, макс, эерэг жилийн тоо, нийт жил (groupby + agg).

    Параметр: df — урт хүснэгт; group_col — бүлэглэх багана; value_col — утга; since — энэ оноос хойш (year >= since).
    Буцаана: DataFrame[group_col, mean, min, max, years_positive, years_total], mean буурахаар.
    """
    d = df if since is None else df[df["year"] >= since]
    pos = (d[value_col] > 0).astype(int)
    tmp = d.assign(_pos=pos)
    out = (tmp.groupby(group_col)
              .agg(mean=(value_col, "mean"), min=(value_col, "min"), max=(value_col, "max"),
                   years_positive=("_pos", "sum"), years_total=(value_col, "count"))
              .reset_index()
              .sort_values("mean", ascending=False)
              .reset_index(drop=True))
    out[["mean", "min", "max"]] = out[["mean", "min", "max"]].round(0)
    return out


def consistency_checks(long_df, totals, indicator_col, name_col, total_name, in_label, out_label,
                       total_names, region_of, name_out="aimag", n_expected=None):
    """Үр дүнгийн автомат шалгалтууд (OK / FAIL).

    Шалгана: оны бүрэн байдал; нутаг+он давтагдахгүй; хоёр урсгал зэрэг бий/зэрэг байхгүй;
    нутгуудын нийлбэр = улсын дүн (он, урсгал бүр); бүсийн аймгуудын нийлбэр = бүсийн дүн;
    дүнгийн нэр урт хүснэгтэд байхгүй; нутгийн тоо.
    Буцаана: (checks: dict{нэр: bool}, details: dict{"national_mismatch": DataFrame, "region_mismatch": list}).
    """
    year_cols = [c for c in totals.columns if str(c).isdigit()]
    checks = {}
    years_per = long_df.groupby(name_out)["year"].nunique()
    checks["Нутаг бүрт бүх он байна"] = bool((years_per == len(year_cols)).all())
    checks["aimag+year түлхүүр давтагдахгүй"] = bool(long_df.duplicated(subset=[name_out, "year"]).sum() == 0)
    one_only = long_df["in_mig"].isnull() != long_df["out_mig"].isnull()
    checks["Хоёр урсгал зэрэг бий эсвэл зэрэг байхгүй"] = bool(one_only.sum() == 0)

    nat = national_balance(totals, indicator_col, name_col, total_name, in_label, out_label)
    s = long_df.groupby("year")[["in_mig", "out_mig"]].sum().reset_index()
    s["official_in"] = s["year"].map(dict(zip(nat["year"], nat["in_total"])))
    s["official_out"] = s["year"].map(dict(zip(nat["year"], nat["out_total"])))
    s["diff_in"] = s["in_mig"] - s["official_in"]
    s["diff_out"] = s["out_mig"] - s["official_out"]
    nat_mismatch = s[(s["diff_in"] != 0) | (s["diff_out"] != 0)]
    checks["Нутгуудын нийлбэр = улсын дүн (бүх он, хоёр урсгал)"] = bool(len(nat_mismatch) == 0)

    region_mismatch = []
    for region in [r for r in total_names if r != total_name]:
        members = [a for a, r in region_of.items() if r == region]
        for label, col in [(in_label, "in_mig"), (out_label, "out_mig")]:
            official = totals[(totals[name_col] == region) & (totals[indicator_col] == label)][year_cols].iloc[0]
            ours = long_df[long_df[name_out].isin(members)].groupby("year")[col].sum()
            for y in year_cols:
                if ours[int(y)] != official[y]:
                    region_mismatch.append((region, label, int(y), int(ours[int(y)] - official[y])))
    checks["Бүсийн аймгуудын нийлбэр = бүсийн дүн"] = bool(len(region_mismatch) == 0)
    checks["Дүнгийн мөр урт хүснэгтэд ороогүй"] = bool(not long_df[name_out].isin(total_names).any())
    if n_expected is not None:
        checks[f"Яг {n_expected} нутаг"] = bool(long_df[name_out].nunique() == n_expected)
    return checks, {"national_mismatch": nat_mismatch, "region_mismatch": region_mismatch}
