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

Preparing and pushing to GitHub

If you want to create a new GitHub repository and push this project, here are two simple options.

Option A — using the `gh` CLI (recommended if installed):

```bash
gh repo create <OWNER>/<REPO> --public --source=. --remote=origin --push
```

Option B — manual steps (works everywhere):

```bash
git init
git add --all
git commit -m "Initial import: processed datasets and scripts"
# create a repo on GitHub via web and copy the remote URL, or use your ssh path
git remote add origin git@github.com:<OWNER>/<REPO>.git
git branch -M main
git push -u origin main
```

Notes before pushing:
- The repo contains processed CSVs under `data/`. If you prefer not to store large data in Git, consider removing or moving large files to `data/originals/` and using Git LFS for big binaries.
- `.gitignore` already excludes common virtualenv and large binary patterns; review it before push.

If you want, I can initialize the local git repo and make the first commit for you, or attempt to create the GitHub repo using `gh` if you authorize it. Which do you want me to do?

