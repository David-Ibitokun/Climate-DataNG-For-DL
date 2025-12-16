"""
validate_datasets.py

Performs checks:
- Year range in FNN features (expect 1990-2023)
- Regions present in annual/regional files (expect 6 regions)
- No missing yields in crop file
- Monthly sequence completeness: 12 months per (Year,Region)

Run: python validate_datasets.py
"""
from pathlib import Path
import pandas as pd
import sys


def load_csv(p):
    p = Path(p)
    if not p.exists():
        print(f"MISSING: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"ERROR reading {p}: {e}")
        return None


base = Path.cwd() / 'data'
fnn_p = base / 'fnn_dataset' / 'processed' / 'fnn_features.csv'
annual_p = base / 'fnn_dataset' / 'processed' / 'regional_annual_climate_features.csv'
monthly_p = base / 'lstm_dataset' / 'raw' / 'regional_monthly_climate_nigeria.csv'
crops_p = base / 'fnn_dataset' / 'raw' / 'nigeria_crop_yields.csv'

expected_years = list(range(1990, 2024))
expected_regions = ['North Central','North East','North West','South East','South South','South West']

issues = []

# Load files
fnn = load_csv(fnn_p)
annual = load_csv(annual_p)
monthly = load_csv(monthly_p)
crops = load_csv(crops_p)

print('\n=== Quick file existence ===')
for p in [fnn_p, annual_p, monthly_p, crops_p]:
    print(p, 'OK' if Path(p).exists() else 'MISSING')

if fnn is not None:
    yrs = sorted(fnn['Year'].dropna().astype(int).unique()) if 'Year' in fnn.columns else []
    print('\nFNN features: years', yrs[0] if yrs else None, 'to', yrs[-1] if yrs else None)
    # check expected range coverage
    miss_years = [y for y in expected_years if y not in yrs]
    if miss_years:
        issues.append(f'FNN missing years: {miss_years}')
    # check crop columns
    crop_cols = [c for c in fnn.columns if c.startswith('yield_')]
    print('FNN crop columns:', crop_cols)
    if crop_cols:
        for c in crop_cols:
            nmiss = fnn[c].isna().sum()
            if nmiss>0:
                issues.append(f'FNN column {c} has {nmiss} missing values')

if annual is not None:
    yrs_a = sorted(annual['Year'].dropna().astype(int).unique()) if 'Year' in annual.columns else []
    regs = sorted(annual['Region'].dropna().unique()) if 'Region' in annual.columns else []
    print('\nAnnual aggregated climate: years', yrs_a[0] if yrs_a else None, 'to', yrs_a[-1] if yrs_a else None)
    print('Regions in annual file:', regs)
    # check regions
    missing_regions = [r for r in expected_regions if r not in regs]
    if missing_regions:
        issues.append(f'Annual file missing regions: {missing_regions}')
    # expected rows
    expected_rows = len(expected_years) * len(expected_regions)
    actual_rows = len(annual)
    print('Annual rows:', actual_rows, 'expected', expected_rows)
    if actual_rows != expected_rows:
        issues.append(f'Annual row count {actual_rows} != expected {expected_rows}')

if monthly is not None:
    # Ensure Year, Month, Region
    for col in ['Year','Month','Region']:
        if col not in monthly.columns:
            issues.append(f'Monthly file missing column: {col}')
    if all(c in monthly.columns for c in ['Year','Month','Region']):
        # count months per (Year,Region)
        grp = monthly.groupby(['Year','Region'])['Month'].nunique().reset_index(name='months')
        bad = grp[grp['months']!=12]
        if not bad.empty:
            issues.append(f'{len(bad)} year-region combos have !=12 months (sample shown)')
            print('\nSample missing-month entries (Year,Region,months):')
            print(bad.head().to_string(index=False))
        else:
            print('\nMonthly sequences: all year-region combos have 12 months')

if crops is not None:
    # check crops present
    crops_present = sorted(crops['Crop'].unique()) if 'Crop' in crops.columns else []
    print('\nCrops present:', crops_present)
    # ensure for each crop, years 1990-2023 present and no missing Yield
    for crop in crops_present:
        dfc = crops[crops['Crop']==crop]
        yrs_c = sorted(dfc['Year'].dropna().astype(int).unique())
        missing_y = [y for y in expected_years if y not in yrs_c]
        if missing_y:
            issues.append(f'Crop {crop} missing years: {missing_y}')
        nmiss = dfc['Yield'].isna().sum()
        if nmiss>0:
            issues.append(f'Crop {crop} has {nmiss} missing yields')

print('\n=== Validation summary ===')
if issues:
    for it in issues:
        print('- ISSUE:', it)
    print('\nPlease review the issues above.')
    sys.exit(2)
else:
    print('All checks passed: year ranges, regions, crop yields, and monthly sequences appear complete.')
    sys.exit(0)
