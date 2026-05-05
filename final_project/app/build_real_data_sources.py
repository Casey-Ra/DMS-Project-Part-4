from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


PART4_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PART4_ROOT / "data_raw"
MODEL_DIR = PART4_ROOT / "model"

BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"
CMS_RATE_PUF_URL = "https://download.cms.gov/marketplace-puf/2024/rate-puf.zip"

BRFSS_ZIP = RAW_DIR / "LLCP2023XPT.zip"
CMS_RATE_ZIP = RAW_DIR / "rate-puf-2024.zip"

BRFSS_OUTPUT = MODEL_DIR / "brfss_lifestyle_risk_training.csv"
CMS_OUTPUT = MODEL_DIR / "cms_rate_benchmark.csv"

BRFSS_COLUMNS = [
    "_AGE80",
    "_SMOKER3",
    "_BMI5CAT",
    "_TOTINDA",
    "_RFBING6",
    "_RFDRHV8",
    "DIABETE4",
    "GENHLTH",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build real Centers for Disease Control and Prevention (CDC) BRFSS "
            "training data and Centers for Medicare & Medicaid Services (CMS) "
            "rate benchmarks."
        )
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def download_if_needed(url: str, destination: Path, force_download: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force_download:
        print(f"Using cached file: {destination}")
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved: {destination}")


def valid_values(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[BRFSS_COLUMNS].copy()
    clean = clean.dropna()

    clean = clean[
        clean["_AGE80"].between(18, 80)
        & clean["_SMOKER3"].isin([1, 2, 3, 4])
        & clean["_BMI5CAT"].isin([1, 2, 3, 4])
        & clean["_TOTINDA"].isin([1, 2])
        & clean["_RFBING6"].isin([1, 2])
        & clean["_RFDRHV8"].isin([1, 2])
        & clean["DIABETE4"].isin([1, 3, 4])
        & clean["GENHLTH"].isin([1, 2, 3, 4, 5])
    ]
    return clean


def transform_brfss(frame: pd.DataFrame) -> pd.DataFrame:
    clean = valid_values(frame)
    transformed = pd.DataFrame(
        {
            "age": clean["_AGE80"].astype(int),
            "tobacco_user": clean["_SMOKER3"].isin([1, 2]).astype(int),
            "obese": (clean["_BMI5CAT"] == 4).astype(int),
            "physical_inactivity": (clean["_TOTINDA"] == 2).astype(int),
            "binge_drinking": (clean["_RFBING6"] == 2).astype(int),
            "heavy_drinking": (clean["_RFDRHV8"] == 2).astype(int),
            "diabetes": (clean["DIABETE4"] == 1).astype(int),
            "general_health": clean["GENHLTH"].astype(int),
        }
    )

    transformed["risk_points"] = (
        (transformed["age"] >= 55).astype(int)
        + transformed["tobacco_user"]
        + transformed["obese"]
        + transformed["physical_inactivity"]
        + transformed["binge_drinking"]
        + transformed["heavy_drinking"]
        + transformed["diabetes"] * 2
        + (transformed["general_health"] >= 4).astype(int)
    )
    transformed["high_risk"] = (transformed["risk_points"] >= 3).astype(int)
    return transformed


def build_brfss_training_data() -> None:
    pieces = []
    row_count = 0

    with zipfile.ZipFile(BRFSS_ZIP) as archive:
        xpt_name = archive.namelist()[0]
        with archive.open(xpt_name) as xpt_file:
            reader = pd.read_sas(
                xpt_file,
                format="xport",
                iterator=True,
                chunksize=50000,
            )
            for chunk in reader:
                transformed = transform_brfss(chunk)
                if transformed.empty:
                    continue
                pieces.append(transformed)
                row_count += len(transformed)

    if not pieces:
        raise RuntimeError("No valid BRFSS rows were transformed.")

    dataset = pd.concat(pieces, ignore_index=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(BRFSS_OUTPUT, index=False)

    print("Centers for Disease Control and Prevention (CDC) BRFSS lifestyle training data built")
    print("-----------------------------------------------------------------------------------")
    print(f"Rows written: {len(dataset)}")
    print(f"High-risk rows: {int(dataset['high_risk'].sum())}")
    print(f"Saved: {BRFSS_OUTPUT}")


def build_cms_rate_benchmark() -> None:
    use_columns = ["Age", "IndividualRate", "IndividualTobaccoRate"]
    pieces = []

    with zipfile.ZipFile(CMS_RATE_ZIP) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as csv_file:
            for chunk in pd.read_csv(csv_file, usecols=use_columns, chunksize=250000):
                chunk = chunk[pd.to_numeric(chunk["Age"], errors="coerce").notna()].copy()
                chunk["age"] = chunk["Age"].astype(int)
                chunk = chunk[chunk["age"].between(18, 64)]
                chunk["IndividualRate"] = pd.to_numeric(
                    chunk["IndividualRate"], errors="coerce"
                )
                chunk["IndividualTobaccoRate"] = pd.to_numeric(
                    chunk["IndividualTobaccoRate"], errors="coerce"
                )
                pieces.append(
                    chunk[["age", "IndividualRate", "IndividualTobaccoRate"]]
                )

    rates = pd.concat(pieces, ignore_index=True)
    benchmark = (
        rates.groupby("age", as_index=False)
        .agg(
            cms_base_monthly_rate=("IndividualRate", "median"),
            cms_tobacco_monthly_rate=("IndividualTobaccoRate", "median"),
        )
        .round(2)
    )
    benchmark["cms_tobacco_monthly_rate"] = benchmark[
        "cms_tobacco_monthly_rate"
    ].fillna(benchmark["cms_base_monthly_rate"])
    benchmark.to_csv(CMS_OUTPUT, index=False)

    print()
    print("Centers for Medicare & Medicaid Services (CMS) Rate PUF benchmark built")
    print("-----------------------------------------------------------------------")
    print(f"Ages written: {len(benchmark)}")
    print(f"Saved: {CMS_OUTPUT}")


def main() -> None:
    args = parse_args()
    download_if_needed(BRFSS_URL, BRFSS_ZIP, args.force_download)
    download_if_needed(CMS_RATE_PUF_URL, CMS_RATE_ZIP, args.force_download)
    build_brfss_training_data()
    build_cms_rate_benchmark()


if __name__ == "__main__":
    main()
