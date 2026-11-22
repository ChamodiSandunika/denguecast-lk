"""
download_data.py
Downloads the Sri Lanka weekly dengue dataset and saves it as CSV files.

Source data : Weekly Epidemiological Reports (WER), Epidemiology Unit,
              Ministry of Health, Sri Lanka.
Tidy version: `srilanka_weekly_data` in the denguedatahub R package
              (Talagala & co., https://github.com/thiyangt/denguedatahub).

Run once:  python download_data.py
Outputs  :  data/dengue_weekly_raw.csv    (as published, 2006-2026)
            data/dengue_weekly_clean.csv  (cleaned, used by the notebook)
"""
import os
import urllib.request

import pandas as pd
import pyreadr

URL = ("https://raw.githubusercontent.com/thiyangt/denguedatahub/main/"
       "data/srilanka_weekly_data.rda")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def download(path_rda: str) -> None:
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, path_rda)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # Dates come in two formats (MM/DD/YYYY before 2022, ISO after)
    df["start_date"] = pd.to_datetime(df["start.date"], format="mixed")
    df["end_date"] = pd.to_datetime(df["end.date"], format="mixed")
    df = df.rename(columns={"week": "epi_week"})
    df["year"] = df["year"].astype(int)
    df["epi_week"] = df["epi_week"].astype(int)
    # Kalmunai is a separate RDHS division inside Ampara district
    df["district"] = df["district"].replace({"Kalmune": "Kalmunai",
                                             "Hambanthota": "Hambantota",
                                             "NuwaraEliya": "Nuwara Eliya"})
    # The series starts with a single week in 2006 -> drop it
    df = df[df["start_date"] >= "2006-12-30"]

    # The last weeks of the WER series are provisional: late reports are added
    # later. From the week starting 17 Aug 2026 the counts collapse in a way that is
    # not possible epidemiologically (Gampaha 567 -> 48, Kandy 288 -> 10 in one week),
    # so these incomplete weeks are removed.
    PROVISIONAL_FROM = pd.Timestamp("2026-08-17")
    n_prov = df["start_date"].ge(PROVISIONAL_FROM).sum()
    df = df[df["start_date"] < PROVISIONAL_FROM]
    print(f"Removed {n_prov} provisional rows (weeks from {PROVISIONAL_FROM.date()})")

    # Build a complete (week x division) grid so that no week is missing
    weeks = (df[["start_date", "end_date", "year", "epi_week"]]
             .drop_duplicates("start_date").sort_values("start_date"))
    grid = weeks.merge(pd.DataFrame({"district": sorted(df["district"].unique())}),
                       how="cross")
    out = grid.merge(df[["start_date", "district", "cases"]],
                     on=["start_date", "district"], how="left")
    n_missing = out["cases"].isna().sum()
    out = out.sort_values(["district", "start_date"])
    out["cases"] = (out.groupby("district")["cases"]
                    .transform(lambda s: s.interpolate(limit_direction="both"))
                    .round().astype(int))
    print(f"Filled {n_missing} missing district-week value(s) by interpolation")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    rda = os.path.join(DATA, "srilanka_weekly_data.rda")
    if not os.path.exists(rda):
        download(rda)
    raw = list(pyreadr.read_r(rda).values())[0]
    raw.to_csv(os.path.join(DATA, "dengue_weekly_raw.csv"), index=False)
    clean_df = clean(raw)
    clean_df.to_csv(os.path.join(DATA, "dengue_weekly_clean.csv"), index=False)
    print("Raw rows:", len(raw), "| Clean rows:", len(clean_df),
          "| Divisions:", clean_df["district"].nunique(),
          "| Weeks:", clean_df["start_date"].nunique(),
          "| Range:", clean_df["start_date"].min().date(), "to",
          clean_df["start_date"].max().date())
