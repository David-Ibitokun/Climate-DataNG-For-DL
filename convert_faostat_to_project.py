"""
convert_faostat_to_project.py

Reads a FAOSTAT CSV (downloaded by the user) and writes the project's
`data/fnn_dataset/raw/nigeria_crop_yields.csv` in the simple Year,Crop,Yield format.

Usage: python convert_faostat_to_project.py
"""
from pathlib import Path
import pandas as pd


def main():
    workspace = Path(__file__).resolve().parent
    # Default input file (from your attachments)
    fao_in = workspace / 'FAOSTAT_data_en_12-16-2025.csv'
    out_dir = workspace / 'data' / 'fnn_dataset' / 'raw'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'nigeria_crop_yields.csv'

    if not fao_in.exists():
        raise FileNotFoundError(f"FAOSTAT file not found: {fao_in}")

    df = pd.read_csv(fao_in)

    # Keep only the three crops we want and map names to project conventions
    keep_items = {
        'Maize (corn)': 'Maize',
        'Cassava': 'Cassava',
        'Yams': 'Yam'
    }

    df = df[df['Item'].isin(keep_items.keys())].copy()

    # Extract Year, Item, Value -> Year, Crop, Yield
    df['Crop'] = df['Item'].map(keep_items)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df['Yield'] = pd.to_numeric(df['Value'], errors='coerce')

    out_df = df[['Year','Crop','Yield']].sort_values(['Crop','Year'])

    # Save in simple long format expected by the project
    out_df.to_csv(out_file, index=False)
    print(f"Wrote {out_file} ({len(out_df)} rows)")


if __name__ == '__main__':
    main()
