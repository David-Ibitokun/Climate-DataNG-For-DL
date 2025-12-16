**Dataset README**

- **Purpose:** Brief provenance and data-cleaning notes for the project's `data/` folder.

- **FAOSTAT import:**
  - Source file: `FAOSTAT_data_en_12-16-2025.csv` (downloaded by user from FAO).
  - Imported into project as `data/fnn_dataset/raw/nigeria_crop_yields.csv` and copied to `data/hybrid_dataset/fnn_dataset/raw/nigeria_crop_yields.csv`.
  - Contains national yields (kg/ha) for Cassava, Maize, and Yams for years 1990–2023. These values are authoritative FAO figures (flags preserved in original FAO CSV).

- **Month=13 cleanup (NASA POWER monthly):**
  - The raw NASA POWER monthly point export included an extra row per year with `Month==13` (this row represents yearly aggregates or metadata produced by the API export and is not a calendar month).
  - I removed all rows where `Month` is outside the 1–12 range from `data/lstm_dataset/raw/regional_monthly_climate_nigeria.csv` to ensure each (Year,Region) has exactly 12 months.
  - The cleanup script added: `fix_monthly_filter.py` (filters `Month` to 1..12). This change was applied and outputs were regenerated.

- **CO₂ data:**
  - Source: Our World in Data CO₂ dataset (used as a documented fallback when World Bank indicator was unavailable).
  - File in project: `data/fnn_dataset/raw/nigeria_co2_emissions.csv` (units converted as documented in scripts).

- **Processed/Generated files (deterministic):**
  - `data/fnn_dataset/processed/fnn_features.csv` — aggregated national/regional features merged with CO₂ and FAO yields.
  - `data/fnn_dataset/processed/regional_annual_climate_features.csv` — annual aggregates by region.
  - `data/lstm_dataset/sequences/lstm_monthly_sequences.csv` — cleaned monthly sequences (12 months per year-region).
  - Hybrid copies available under `data/hybrid_dataset/` (exact copies of processed/raw files used for hybrid model inputs).

- **Notes & reproducibility:**
  - Conversion/import script: `convert_faostat_to_project.py` (turns FAOSTAT CSV into the project's Year,Crop,Yield format).
  - Processing script: `process_data_now.py` (regenerates processed datasets from raw inputs).
  - Validation script: `validate_datasets.py` (checks year ranges, regions, missing yields, and monthly completeness).
  - The month=13 cleanup is intentional and documented above; nothing was fabricated — FAOSTAT rows are the user-supplied official file.

If you want me to commit these files to version control or add further provenance fields (e.g., original FAO flag column extraction), tell me which you prefer.
