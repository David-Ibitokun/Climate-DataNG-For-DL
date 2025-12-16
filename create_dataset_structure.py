"""
create_dataset_structure.py

Creates an idempotent dataset folder structure for LSTM, FNN and Hybrid models.

Usage: run directly with Python. Safe to run multiple times.

Folder purposes (comments below are included near the creation logic):
 - raw/: Original downloaded datasets (no modification)
 - processed/: Cleaned, normalized, aggregated data
 - sequences/: Time-series sequences prepared for LSTM
 - metadata/: Feature descriptions, scaling info, README files
"""

from pathlib import Path
import os


def create_structure(base_dir: Path = Path("data")) -> None:
    """Create the dataset directory tree exactly as specified.

    The function is idempotent: it will not raise if directories already exist.
    It prints confirmation messages for each folder created or already present.
    """

    # Define the exact directory tree to create
    dirs = [
        # FNN dataset
        base_dir / "fnn_dataset" / "raw",       # raw/ -> Original downloaded datasets (no modification)
        base_dir / "fnn_dataset" / "processed", # processed/ -> Cleaned, normalized, aggregated data
        base_dir / "fnn_dataset" / "metadata",  # metadata/ -> Feature descriptions, scaling info, README files

        # LSTM dataset
        base_dir / "lstm_dataset" / "raw",      # raw/
        base_dir / "lstm_dataset" / "sequences",# sequences/ -> Time-series sequences prepared for LSTM
        base_dir / "lstm_dataset" / "metadata", # metadata/

        # Hybrid dataset contains nested LSTM and FNN datasets
        base_dir / "hybrid_dataset" / "lstm_dataset" / "raw",
        base_dir / "hybrid_dataset" / "lstm_dataset" / "sequences",
        base_dir / "hybrid_dataset" / "lstm_dataset" / "metadata",

        base_dir / "hybrid_dataset" / "fnn_dataset" / "raw",
        base_dir / "hybrid_dataset" / "fnn_dataset" / "processed",
        base_dir / "hybrid_dataset" / "fnn_dataset" / "metadata",
    ]

    # Create directories idempotently and print confirmations
    for d in dirs:
        try:
            if d.exists():
                print(f"Exists: {d}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                print(f"Created: {d}")
        except Exception as e:
            # Do not delete or modify existing files; just report the error
            print(f"Error creating {d}: {e}")

    print("\nDataset folder structure ready.")


if __name__ == "__main__":
    # Run against the current working directory so script is portable
    base = Path(os.getcwd()) / "data"
    create_structure(base)
