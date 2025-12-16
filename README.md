# Climate-DataNG-For-DL

This repository collects and preprocesses climate (NASA POWER), CO₂ (OWID fallback),
and FAOSTAT crop-yield data for Nigeria (1990–2023) and produces model-ready
datasets for LSTM, FNN and hybrid FNN–LSTM experiments.

Contents (high-level):

- `data/` — raw and processed datasets (see `data/README.md` and `datasource.md` for provenance)
- `process_data_now.py` — main preprocessing pipeline that generates the processed CSVs
- `convert_faostat_to_project.py` — importer for user FAOSTAT CSVs
- `fix_monthly_filter.py` — removes extraneous `Month==13` rows from NASAPOWER exports
- `validate_datasets.py` — runs integrity checks (year range, regions, sequences)
- `documentation.md` — quick workflow and structure

Prerequisites

- Python 3.8+ and `pip`
- Recommended packages: `pandas`, `numpy`, `requests`

Quick start (local)

1. Convert FAOSTAT (if you have it):

```bash
python convert_faostat_to_project.py
```

2. Clean NASA POWER monthly export (if needed):

```bash
python fix_monthly_filter.py
```

3. Generate processed datasets:

```bash
python process_data_now.py
```

4. Validate outputs:

```bash
python validate_datasets.py
```
