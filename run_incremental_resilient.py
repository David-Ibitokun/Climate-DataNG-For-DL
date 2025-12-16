from pathlib import Path
import time
import pandas as pd
from setup_data_project import REGIONS, nasa_power_monthly, aggregate_annual_climate, fetch_worldbank_co2, ingest_faostat_user_csv


def main():
    base = Path.cwd() / "data"
    monthly_path = base / "lstm_dataset" / "raw" / "regional_monthly_climate_nigeria.csv"
    monthly_path.parent.mkdir(parents=True, exist_ok=True)

    for region, coords in REGIONS.items():
        lat, lon = coords
        print(f"[resilient] Fetching region: {region} ({lat},{lon})")
        try:
            # use shorter connect/read timeouts and fewer retries to avoid long blocks
            df = nasa_power_monthly(lat, lon, retries=2, timeout=(5, 10))
            df["Region"] = region
            cols = ["Region", "Year", "Month", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
            df = df[cols]

            if monthly_path.exists():
                existing = pd.read_csv(monthly_path)
                combined = pd.concat([existing, df], ignore_index=True, sort=False)
                combined = combined.drop_duplicates(subset=["Region", "Year", "Month"])
            else:
                combined = df

            combined.to_csv(monthly_path, index=False)
            print(f"Saved monthly CSV ({len(combined)} rows) after region {region}")

            agg_path = base / "fnn_dataset" / "processed" / "regional_annual_climate_features.csv"
            aggregate_annual_climate(combined, agg_path)

        except Exception as e:
            print(f"[resilient] Error fetching {region}: {e}")

        time.sleep(0.5)

    # fetch co2 and ingest faostat
    co2_path = base / "fnn_dataset" / "raw" / "nigeria_co2_emissions.csv"
    print("[resilient] Fetching World Bank CO2...")
    fetch_worldbank_co2(co2_path)

    user_crop = Path.cwd() / "nigeria_crop_yields.csv"
    crops_dest = base / "fnn_dataset" / "raw" / "nigeria_crop_yields.csv"
    ingest_faostat_user_csv(user_crop, crops_dest)

    print("[resilient] Incremental resilient fetch complete.")


if __name__ == "__main__":
    main()
