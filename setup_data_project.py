"""
setup_data_project.py

Creates the data folder structure, downloads Nigeria regional climate (NASA POWER),
World Bank CO2, and ingests a user-provided FAOSTAT crop yields CSV. Saves raw
CSVs into the dataset folders and generates a Jupyter notebook `data_processing.ipynb`
that contains data cleaning, aggregation, and preprocessing steps.

Dependencies: requests, pandas, numpy

Run:
    python setup_data_project.py [--quick]

Options:
    --quick    : run a short test (single region, short sleeps)
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List

import requests
import pandas as pd
import numpy as np

# Configuration
START_YEAR = 1990
END_YEAR = 2023

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_VARS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
WB_CO2_INDICATOR = "EN.ATM.CO2E.KT"
FAO_CROPS = ["Maize", "Cassava", "Yam"]

# Nigeria regions central coordinates
REGIONS: Dict[str, Tuple[float, float]] = {
    "North Central": (8.5, 7.0),
    "North East": (11.8, 13.1),
    "North West": (12.0, 7.5),
    "South East": (6.0, 7.5),
    "South South": (5.0, 6.0),
    "South West": (6.5, 3.5),
}


def ensure_structure(base: Path = Path("data")) -> None:
    """Create the required idempotent folder structure using pathlib/os.

    Purpose comments:
    - raw/: Original downloaded datasets (no modification)
    - processed/: Cleaned, normalized, aggregated data
    - sequences/: Time-series sequences prepared for LSTM
    - metadata/: Feature descriptions, scaling info, README files
    """
    paths = [
        base / "fnn_dataset" / "raw",
        base / "fnn_dataset" / "processed",
        base / "fnn_dataset" / "metadata",
        base / "lstm_dataset" / "raw",
        base / "lstm_dataset" / "sequences",
        base / "lstm_dataset" / "metadata",
        base / "hybrid_dataset" / "lstm_dataset" / "raw",
        base / "hybrid_dataset" / "lstm_dataset" / "sequences",
        base / "hybrid_dataset" / "lstm_dataset" / "metadata",
        base / "hybrid_dataset" / "fnn_dataset" / "raw",
        base / "hybrid_dataset" / "fnn_dataset" / "processed",
        base / "hybrid_dataset" / "fnn_dataset" / "metadata",
    ]

    for p in paths:
        try:
            if p.exists():
                print(f"Exists: {p}")
            else:
                p.mkdir(parents=True, exist_ok=True)
                print(f"Created: {p}")
        except Exception as e:
            print(f"Error creating {p}: {e}")


def nasa_power_monthly(lat: float, lon: float, start=START_YEAR, end=END_YEAR, retries=3, timeout=20) -> pd.DataFrame:
    """Fetch monthly NASA POWER variables for a point, with retries.

    Returns a DataFrame with Year, Month and variables.
    """
    params = {
        "start": str(start),
        "end": str(end),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(NASA_VARS),
        "format": "JSON",
    }

    backoff = 1
    for attempt in range(retries):
        try:
            resp = requests.get(NASA_POWER_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            params_dict = data.get("properties", {}).get("parameter", {})

            ym_keys = set()
            for vardict in params_dict.values():
                ym_keys.update(vardict.keys())

            records: List[dict] = []
            for ym in sorted(ym_keys):
                try:
                    year = int(ym[:4])
                    month = int(ym[4:6])
                except Exception:
                    continue
                row = {"Year": year, "Month": month}
                for var in NASA_VARS:
                    row[var] = params_dict.get(var, {}).get(ym, np.nan)
                records.append(row)
            df = pd.DataFrame.from_records(records)
            for var in NASA_VARS:
                df[var] = pd.to_numeric(df[var], errors="coerce")
            return df
        except Exception as e:
            print(f"NASA POWER attempt {attempt+1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("NASA POWER unavailable after retries")


def gather_climate_for_regions(regions: Dict[str, Tuple[float, float]], out_path: Path, quick: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    regions_to_use = {k: v} if quick and (k := list(regions.keys())[0]) else regions
    for region_name, (lat, lon) in regions_to_use.items():
        print(f"Fetching NASA POWER for {region_name} ({lat},{lon})")
        df = nasa_power_monthly(lat, lon)
        df["Region"] = region_name
        frames.append(df)
        time.sleep(0.1 if quick else 1.0)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    cols = ["Region", "Year", "Month"] + NASA_VARS
    combined = combined[cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved climate CSV to {out_path}")
    return combined


def aggregate_annual_climate(monthly_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    monthly_df["Year"] = monthly_df["Year"].astype(int)
    monthly_df["Region"] = monthly_df["Region"].astype(str)

    agg = monthly_df.groupby(["Region", "Year"]).agg(
        annual_rainfall_mm=("PRECTOTCORR", lambda x: np.nansum(x.values.astype(float))),
        annual_mean_temp_c=("T2M", lambda x: np.nanmean(x.values.astype(float))),
        annual_max_temp_c=("T2M_MAX", lambda x: np.nanmax(x.values.astype(float))),
    ).reset_index()

    # Ensure full coverage per region
    years = list(range(START_YEAR, END_YEAR + 1))
    rows: List[dict] = []
    for region in REGIONS.keys():
        sub = agg[agg["Region"] == region].set_index("Year")
        for y in years:
            if y in sub.index:
                rows.append({"Region": region, "Year": y, "annual_rainfall_mm": sub.loc[y, "annual_rainfall_mm"], "annual_mean_temp_c": sub.loc[y, "annual_mean_temp_c"], "annual_max_temp_c": sub.loc[y, "annual_max_temp_c"]})
            else:
                rows.append({"Region": region, "Year": y, "annual_rainfall_mm": np.nan, "annual_mean_temp_c": np.nan, "annual_max_temp_c": np.nan})

    final = pd.DataFrame.from_records(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"Saved annual climate features to {out_path}")
    return final


def fetch_worldbank_co2(out_path: Path, quick: bool = False) -> pd.DataFrame:
    url = f"http://api.worldbank.org/v2/country/NGA/indicator/{WB_CO2_INDICATOR}"
    params = {"date": f"{START_YEAR}:{END_YEAR}", "format": "json", "per_page": 1000}
    backoff = 1
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            entries = data[1] if isinstance(data, list) and len(data) > 1 else []
            records = []
            for e in entries:
                try:
                    year = int(e.get("date")) if e.get("date") else None
                except Exception:
                    year = None
                val = e.get("value")
                if year is not None:
                    records.append({"Year": year, "CO2_Emissions_kt": (np.nan if val is None else float(val))})
            df = pd.DataFrame.from_records(records)
            if df.empty:
                df = pd.DataFrame({"Year": list(range(START_YEAR, END_YEAR + 1)), "CO2_Emissions_kt": [np.nan] * (END_YEAR - START_YEAR + 1)})
            else:
                years = pd.Series(range(START_YEAR, END_YEAR + 1), name="Year")
                df = years.to_frame().merge(df, on="Year", how="left")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"Saved CO2 CSV to {out_path}")
            return df
        except Exception as e:
            print(f"World Bank attempt {attempt+1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
    # fallback
    df = pd.DataFrame({"Year": list(range(START_YEAR, END_YEAR + 1)), "CO2_Emissions_kt": [np.nan] * (END_YEAR - START_YEAR + 1)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved CO2 CSV template to {out_path}")
    return df


def ingest_faostat_user_csv(src_path: Path, dest_path: Path) -> pd.DataFrame:
    """Read a user-provided FAOSTAT CSV and save a copy to dest_path.raw.

    If src_path does not exist, create an empty template with Year/Crop/Yield rows.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        df = pd.read_csv(src_path)
        # Keep only Year, Crop, Yield columns if present
        cols = [c for c in ["Year", "Crop", "Yield"] if c in df.columns]
        df = df[cols]
        df.to_csv(dest_path, index=False)
        print(f"Copied user FAOSTAT CSV to {dest_path}")
        return df
    else:
        # create template
        rows = []
        for crop in FAO_CROPS:
            for y in range(START_YEAR, END_YEAR + 1):
                rows.append({"Year": y, "Crop": crop, "Yield": np.nan})
        df = pd.DataFrame.from_records(rows)
        df.to_csv(dest_path, index=False)
        print(f"No user FAOSTAT found; created template at {dest_path}")
        return df


def write_metadata_files(base: Path) -> None:
    """Create small metadata files describing features and scaling placeholders."""
    # FNN metadata
    fnn_meta = base / "fnn_dataset" / "metadata"
    lstm_meta = base / "lstm_dataset" / "metadata"
    hybrid_fnn_meta = base / "hybrid_dataset" / "fnn_dataset" / "metadata"
    hybrid_lstm_meta = base / "hybrid_dataset" / "lstm_dataset" / "metadata"

    feature_desc = (
        "Features:\n"
        "- annual_rainfall_mm: total rainfall per year (mm)\n"
        "- annual_mean_temp_c: annual mean temperature (°C)\n"
        "- annual_max_temp_c: annual maximum temperature (°C)\n"
        "- CO2_Emissions_kt: national CO2 emissions (kt)\n"
        "- Yield: crop yield (kg/ha)\n"
    )

    scaling_template = {"feature_scales": {}, "notes": "Fill with mean/std or min/max per feature after preprocessing."}

    for meta in [fnn_meta, lstm_meta, hybrid_fnn_meta, hybrid_lstm_meta]:
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "feature_description.txt").write_text(feature_desc)
        (meta / "scaling_params.json").write_text(json.dumps(scaling_template, indent=2))
        print(f"Wrote metadata to {meta}")


def generate_notebook(nb_path: Path, base: Path) -> None:
    """Generate a Jupyter notebook JSON with preprocessing steps and explanations.

    The notebook loads the raw CSVs, performs cleaning, aggregation, and saves
    processed datasets into the `data/` folders.
    """
    nb = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def md(text: str):
        return {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": [text]}

    def code(lines: List[str]):
        return {"cell_type": "code", "metadata": {"language": "python"}, "source": [l + "\n" for l in lines], "outputs": [], "execution_count": None}

    # Notebook content
    nb["cells"].append(md("# Data Processing for Nigeria: Climate, CO2, and Crop Yields"))
    nb["cells"].append(md("This notebook loads raw CSVs, cleans missing values, aggregates monthly climate to annual features, and writes processed datasets ready for LSTM, FNN and hybrid models."))

    # Load libraries
    nb["cells"].append(code([
        "import pandas as pd",
        "import numpy as np",
        "from pathlib import Path",
        "base = Path('data')",
    ]))

    # Paths
    nb["cells"].append(code([
        "climate_monthly = base / 'lstm_dataset' / 'raw' / 'regional_monthly_climate_nigeria.csv'",
        "climate_annual = base / 'fnn_dataset' / 'processed' / 'regional_annual_climate_features.csv'",
        "co2_csv = base / 'fnn_dataset' / 'raw' / 'nigeria_co2_emissions.csv'",
        "crop_csv = base / 'fnn_dataset' / 'raw' / 'nigeria_crop_yields.csv'",
    ]))

    # Read raw files cell
    nb["cells"].append(code([
        "df_monthly = pd.read_csv(climate_monthly)",
        "df_annual = pd.read_csv(climate_annual) if climate_annual.exists() else pd.DataFrame()",
        "df_co2 = pd.read_csv(co2_csv)",
        "df_crops = pd.read_csv(crop_csv)",
        "print('Monthly', df_monthly.shape)",
        "print('Annual', df_annual.shape)",
        "print('CO2', df_co2.shape)",
        "print('Crops', df_crops.shape)",
    ]))

    # Cleaning and renaming
    nb["cells"].append(md("## Cleaning: ensure numeric types and consistent column names"))
    nb["cells"].append(code([
        "# Standardize column names",
        "df_monthly = df_monthly.rename(columns={'PRECTOTCORR':'rainfall_mm','T2M':'temp_avg_c','T2M_MAX':'temp_max_c','T2M_MIN':'temp_min_c'})",
        "# Ensure numeric",
        "for c in ['rainfall_mm','temp_avg_c','temp_max_c','temp_min_c']:",
        "    if c in df_monthly.columns:",
        "        df_monthly[c] = pd.to_numeric(df_monthly[c], errors='coerce')",
        "df_co2['CO2_Emissions_kt'] = pd.to_numeric(df_co2['CO2_Emissions_kt'], errors='coerce')",
        "# Crop yields column rename if present",
        "if 'Yield' in df_crops.columns:",
        "    df_crops = df_crops.rename(columns={'Yield':'yield_kg_ha'})",
    ]))

    # Aggregation cell
    nb["cells"].append(md("## Aggregation: Monthly -> Annual features per region"))
    nb["cells"].append(code([
        "agg = df_monthly.groupby(['Region','Year']).agg(annual_rainfall_mm=('rainfall_mm', 'sum'), annual_mean_temp_c=('temp_avg_c','mean'), annual_max_temp_c=('temp_max_c','max')).reset_index()",
        "agg.to_csv(climate_annual, index=False)",
        "print('Saved aggregated annual climate to', climate_annual)",
    ]))

    # Save processed FNN features
    nb["cells"].append(md("## Prepare FNN features and save"))
    nb["cells"].append(code([
        "# Example: merge climate annual (regional) with national CO2 by Year",
        "fnn_features = agg.groupby('Year').agg({'annual_rainfall_mm':'mean','annual_mean_temp_c':'mean','annual_max_temp_c':'mean'}).reset_index()",
        "fnn_features = fnn_features.merge(df_co2, on='Year', how='left')",
        "fnn_out = base / 'fnn_dataset' / 'processed' / 'fnn_features.csv'",
        "fnn_out.parent.mkdir(parents=True, exist_ok=True)",
        "fnn_features.to_csv(fnn_out, index=False)",
        "print('Saved FNN features to', fnn_out)",
    ]))

    # LSTM sequences placeholder
    nb["cells"].append(md("## Prepare LSTM sequences (example)"))
    nb["cells"].append(code([
        "# This is a simple example: pivot monthly data into sequences per Region",
        "seq_dir = base / 'lstm_dataset' / 'sequences'",
        "seq_dir.mkdir(parents=True, exist_ok=True)",
        "# Save a CSV of monthly sequences as-is for downstream processing",
        "seq_out = seq_dir / 'lstm_monthly_sequences.csv'",
        "df_monthly.to_csv(seq_out, index=False)",
        "print('Saved LSTM sequences (monthly) to', seq_out)",
    ]))

    # Hybrid: combine example
    nb["cells"].append(md("## Hybrid dataset: combine LSTM and FNN features for hybrid models"))
    nb["cells"].append(code([
        "hybrid_fnn_out = base / 'hybrid_dataset' / 'fnn_dataset' / 'processed' / 'hybrid_fnn_features.csv'",
        "hybrid_fnn_out.parent.mkdir(parents=True, exist_ok=True)",
        "fnn_features.to_csv(hybrid_fnn_out, index=False)",
        "print('Saved hybrid FNN features to', hybrid_fnn_out)",
    ]))

    # Metadata note cell
    nb["cells"].append(md("## Metadata\nFeature descriptions and placeholder scaling parameters are saved in each dataset `metadata/` folder."))

    # Quick inspection
    nb["cells"].append(code([
        "print('FNN features head:')",
        "print(fnn_features.head())",
        "print('Monthly head:')",
        "print(df_monthly.head())",
    ]))

    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(nb, indent=2))
    print(f"Wrote notebook to {nb_path}")


def main(quick: bool = False):
    base = Path.cwd() / "data"
    ensure_structure(base)

    # Paths to save raw CSVs
    climate_monthly_path = base / "lstm_dataset" / "raw" / "regional_monthly_climate_nigeria.csv"
    climate_annual_path = base / "fnn_dataset" / "processed" / "regional_annual_climate_features.csv"
    co2_path = base / "fnn_dataset" / "raw" / "nigeria_co2_emissions.csv"
    crops_dest = base / "fnn_dataset" / "raw" / "nigeria_crop_yields.csv"

    # 1) Fetch climate (may take time)
    try:
        monthly_df = gather_climate_for_regions(REGIONS, climate_monthly_path, quick=quick)
    except Exception as e:
        print(f"Climate fetch failed: {e}")
        monthly_df = pd.DataFrame()

    # 2) Aggregate annual climate
    if not monthly_df.empty:
        aggregate_annual_climate(monthly_df, climate_annual_path)

    # 3) Fetch CO2
    fetch_worldbank_co2(co2_path, quick=quick)

    # 4) Ingest user FAOSTAT CSV if provided at project root
    user_crop = Path.cwd() / 'nigeria_crop_yields.csv'
    ingest_faostat_user_csv(user_crop, crops_dest)

    # 5) Write metadata files
    write_metadata_files(base)

    # 6) Generate notebook
    nb_path = Path.cwd() / 'data_processing.ipynb'
    generate_notebook(nb_path, base)

    print('\nSetup complete. Data saved under', base)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test run (single region, short waits)')
    args = parser.parse_args()
    main(quick=args.quick)
