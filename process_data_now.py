"""
process_data_now.py

Runs preprocessing to produce:
- data/lstm_dataset/sequences/lstm_monthly_sequences.csv
- data/fnn_dataset/processed/fnn_features.csv
- data/hybrid_dataset/... (copies of the above into hybrid folders)

This script mirrors the steps in the notebook and is safe to re-run.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize names
    df = df.rename(columns={
        'PRECTOTCORR': 'rainfall_mm',
        'T2M': 'temp_avg_c',
        'T2M_MAX': 'temp_max_c',
        'T2M_MIN': 'temp_min_c'
    })
    # Drop non-monthly rows (some sources include month==13)
    if 'Month' in df.columns:
        df = df[df['Month'].between(1,12)]
    # Ensure numeric
    for c in ['rainfall_mm','temp_avg_c','temp_max_c','temp_min_c','Year','Month']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def aggregate_annual_from_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    agg = df_monthly.groupby(['Region','Year']).agg(
        annual_rainfall_mm=('rainfall_mm', 'sum'),
        annual_mean_temp_c=('temp_avg_c','mean'),
        annual_max_temp_c=('temp_max_c','max')
    ).reset_index()
    return agg


def build_fnn_features(annual_df: pd.DataFrame, co2_df: pd.DataFrame, crops_df: pd.DataFrame) -> pd.DataFrame:
    # National-level features: mean of regional annual features
    national = annual_df.groupby('Year').agg(
        annual_rainfall_mm=('annual_rainfall_mm','mean'),
        annual_mean_temp_c=('annual_mean_temp_c','mean'),
        annual_max_temp_c=('annual_max_temp_c','mean')
    ).reset_index()

    co2_df = co2_df.rename(columns={'Year':'Year','CO2_Emissions_kt':'CO2_Emissions_kt'})
    fnn = national.merge(co2_df, on='Year', how='left')

    # Pivot crops to wide format
    if not crops_df.empty:
        crops_pivot = crops_df.pivot_table(index='Year', columns='Crop', values='Yield', aggfunc='mean')
        crops_pivot = crops_pivot.rename(columns=lambda x: f'yield_{str(x).lower()}')
        crops_pivot = crops_pivot.reset_index()
        fnn = fnn.merge(crops_pivot, on='Year', how='left')

    # Handle missing values: linear interpolate then fill with mean
    fnn = fnn.sort_values('Year')
    fnn.interpolate(method='linear', limit_direction='both', inplace=True)
    fnn.fillna(fnn.mean(numeric_only=True), inplace=True)
    return fnn


def save_lstm_sequences(df_monthly: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save cleaned monthly time series for each region
    df_monthly.to_csv(out_path, index=False)
    print(f"Saved LSTM monthly sequences to {out_path}")


def save_fnn_features(fnn_df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fnn_df.to_csv(out_path, index=False)
    print(f"Saved FNN features to {out_path}")


def copy_to_hybrid(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    df.to_csv(dest, index=False)
    print(f"Copied {src} to {dest}")


def main():
    base = Path.cwd() / 'data'
    monthly_p = base / 'lstm_dataset' / 'raw' / 'regional_monthly_climate_nigeria.csv'
    annual_p = base / 'fnn_dataset' / 'processed' / 'regional_annual_climate_features.csv'
    co2_p = base / 'fnn_dataset' / 'raw' / 'nigeria_co2_emissions.csv'
    crops_p = base / 'fnn_dataset' / 'raw' / 'nigeria_crop_yields.csv'

    # Load monthly
    if not monthly_p.exists():
        raise FileNotFoundError(f"Monthly climate file not found: {monthly_p}")
    df_monthly = load_monthly(monthly_p)

    # Aggregate annual and save (overwrite)
    annual_df = aggregate_annual_from_monthly(df_monthly)
    annual_p.parent.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(annual_p, index=False)
    print(f"Saved aggregated annual climate to {annual_p}")

    # Load CO2 and crops
    co2_df = pd.read_csv(co2_p) if co2_p.exists() else pd.DataFrame()
    crops_df = pd.read_csv(crops_p) if crops_p.exists() else pd.DataFrame()

    # Build FNN features
    fnn = build_fnn_features(annual_df, co2_df, crops_df)
    fnn_out = base / 'fnn_dataset' / 'processed' / 'fnn_features.csv'
    save_fnn_features(fnn, fnn_out)

    # LSTM sequences: save cleaned monthly
    lstm_seq_out = base / 'lstm_dataset' / 'sequences' / 'lstm_monthly_sequences.csv'
    save_lstm_sequences(df_monthly, lstm_seq_out)

    # Hybrid copies
    hybrid_fnn_out = base / 'hybrid_dataset' / 'fnn_dataset' / 'processed' / 'hybrid_fnn_features.csv'
    hybrid_lstm_out = base / 'hybrid_dataset' / 'lstm_dataset' / 'sequences' / 'lstm_monthly_sequences.csv'
    copy_to_hybrid(fnn_out, hybrid_fnn_out)
    copy_to_hybrid(lstm_seq_out, hybrid_lstm_out)

    print('Processing complete.')


if __name__ == '__main__':
    main()
