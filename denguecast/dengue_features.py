"""
dengue_features.py
Shared feature engineering for the notebook and the Streamlit dashboard.

Task: predict the number of dengue cases in a division for week t,
using only information that is known at the end of week t-1
(i.e. a 1-week-ahead forecast).
"""
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4]
SEASON_ORDER = ["NE monsoon", "1st inter-monsoon", "SW monsoon", "2nd inter-monsoon"]

NUM_FEATURES = [
    "lag1", "lag2", "lag3", "lag4", "lag52",
    "roll_mean4", "roll_mean8", "roll_std4", "trend",
    "nat_lag1", "week_sin", "week_cos",
]
CAT_FEATURES = ["district", "season"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "cases"


def monsoon_season(month: int) -> str:
    """Sri Lanka's four climatic seasons (Department of Meteorology)."""
    if month in (12, 1, 2):
        return "NE monsoon"
    if month in (3, 4):
        return "1st inter-monsoon"
    if month in (5, 6, 7, 8, 9):
        return "SW monsoon"
    return "2nd inter-monsoon"          # Oct, Nov


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: start_date, epi_week, district, cases (one row per district-week).
    Count features are log1p-transformed (trees do not care, linear/ANN models do)."""
    d = df.sort_values(["district", "start_date"]).copy()
    d["start_date"] = pd.to_datetime(d["start_date"])
    lc = np.log1p(d["cases"])
    g = lc.groupby(d["district"])
    for k in LAGS:
        d[f"lag{k}"] = g.shift(k)
    d["lag52"] = g.shift(52)
    shifted = g.shift(1)
    sg = shifted.groupby(d["district"])
    d["roll_mean4"] = sg.transform(lambda s: s.rolling(4).mean())
    d["roll_mean8"] = sg.transform(lambda s: s.rolling(8).mean())
    d["roll_std4"] = sg.transform(lambda s: s.rolling(4).std())
    d["trend"] = d["lag1"] - d["lag2"]

    nat = d.groupby("start_date")["cases"].sum().sort_index()
    nat_lag1 = np.log1p(nat.shift(1))
    d["nat_lag1"] = d["start_date"].map(nat_lag1)

    w = d["epi_week"].clip(upper=52)
    d["week_sin"] = np.sin(2 * np.pi * w / 52)
    d["week_cos"] = np.cos(2 * np.pi * w / 52)
    d["season"] = d["start_date"].dt.month.map(monsoon_season)
    return d


def make_row(district, epi_week, month, last4, same_week_last_year,
             weeks5to8_mean, national_last_week):
    """Build one feature row from raw case counts (used by the dashboard form).
    last4 = [cases t-1, t-2, t-3, t-4] (most recent first)."""
    l = np.log1p(np.array(last4, dtype=float))
    l58 = np.log1p(float(weeks5to8_mean))
    w = min(int(epi_week), 52)
    row = {
        "lag1": l[0], "lag2": l[1], "lag3": l[2], "lag4": l[3],
        "lag52": np.log1p(same_week_last_year),
        "roll_mean4": l.mean(),
        "roll_mean8": (l.sum() + 4 * l58) / 8,
        "roll_std4": l.std(ddof=1),
        "trend": l[0] - l[1],
        "nat_lag1": np.log1p(national_last_week),
        "week_sin": np.sin(2 * np.pi * w / 52),
        "week_cos": np.cos(2 * np.pi * w / 52),
        "district": district,
        "season": monsoon_season(int(month)),
    }
    return pd.DataFrame([row])[FEATURES]


def risk_level(pred, thresholds_row):
    """Map a forecast to a risk band using the division's own history
    (percentiles of weekly cases in the training period)."""
    if pred >= thresholds_row["p95"]:
        return "Very high"
    if pred >= thresholds_row["p75"]:
        return "High"
    if pred >= thresholds_row["p50"]:
        return "Moderate"
    return "Low"
