# Project Documentation

This short documentation explains the dataset layout, main scripts, and quick reproduction steps.

Structure (key folders):

- `data/`
  - `fnn_dataset/raw/` — raw CO₂ and crop yields
  - `fnn_dataset/processed/` — `fnn_features.csv`, `regional_annual_climate_features.csv`
  - `lstm_dataset/raw/` — `regional_monthly_climate_nigeria.csv`
  - `lstm_dataset/sequences/` — `lstm_monthly_sequences.csv`
  - `hybrid_dataset/` — copies for hybrid experiments

- `scripts/` (root): processing and helper scripts such as `process_data_now.py`, `convert_faostat_to_project.py`, `fix_monthly_filter.py`, `validate_datasets.py`.

Quick workflow:

1. Place FAOSTAT CSV (if you have it) at project root: `FAOSTAT_data_en_12-16-2025.csv`.
2. Convert FAOSTAT into project format:

```bash
python convert_faostat_to_project.py
```

3. Clean monthly NASA POWER export (remove non-calendar months):

```bash
python fix_monthly_filter.py
```

4. Generate processed datasets:

```bash
python process_data_now.py
```

5. Validate outputs:

```bash
python validate_datasets.py
```

Notes:

- `process_data_now.py` overwrites processed files; safe to re-run after raw updates.
- The FAOSTAT import preserves official FAO values; flags are available in the original FAOSTAT CSV if you need them.
- If you prefer to use World Bank CO₂ exports, replace `data/fnn_dataset/raw/nigeria_co2_emissions.csv` and re-run `process_data_now.py`.

Contact:

For further modifications (e.g., state-level disaggregation, additional climate variables, or scaling parameters), ask me to add scripts or examples.
