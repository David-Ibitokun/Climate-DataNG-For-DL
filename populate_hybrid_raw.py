from pathlib import Path


def main():
    base = Path.cwd() / "data"
    copies = [
        (base / "lstm_dataset" / "raw" / "regional_monthly_climate_nigeria.csv",
         base / "hybrid_dataset" / "lstm_dataset" / "raw" / "regional_monthly_climate_nigeria.csv"),
        (base / "fnn_dataset" / "raw" / "nigeria_co2_emissions.csv",
         base / "hybrid_dataset" / "fnn_dataset" / "raw" / "nigeria_co2_emissions.csv"),
        (base / "fnn_dataset" / "raw" / "nigeria_crop_yields.csv",
         base / "hybrid_dataset" / "fnn_dataset" / "raw" / "nigeria_crop_yields.csv"),
    ]

    for src, dst in copies:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"Source missing, skipping: {src}")
                continue
            # copy file contents
            dst.write_bytes(src.read_bytes())
            print(f"Copied {src} -> {dst}")
        except Exception as e:
            print(f"Error copying {src} -> {dst}: {e}")

    print("Hybrid raw population complete.")


if __name__ == '__main__':
    main()
