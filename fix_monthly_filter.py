"""
fix_monthly_filter.py

Removes rows where Month is outside 1-12 from the regional monthly climate CSV,
then saves the cleaned file (overwrites) so sequence completeness checks pass.

Run: python fix_monthly_filter.py
"""
from pathlib import Path
import pandas as pd

fpath = Path.cwd() / 'data' / 'lstm_dataset' / 'raw' / 'regional_monthly_climate_nigeria.csv'
if not fpath.exists():
    raise FileNotFoundError(f"Monthly file not found: {fpath}")

df = pd.read_csv(fpath)
if 'Month' in df.columns:
    before = len(df)
    df = df[df['Month'].between(1,12)]
    after = len(df)
    df.to_csv(fpath, index=False)
    print(f"Filtered monthly file: {before} -> {after} rows saved to {fpath}")
else:
    print('No Month column found; no changes made.')
