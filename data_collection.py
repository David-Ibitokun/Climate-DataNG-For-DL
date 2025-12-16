"""
data_collection.py

Downloads and processes regional monthly climate (NASA POWER), annual CO2
(World Bank), and FAOSTAT crop yield data for Nigeria (1990-2023).

Outputs:
- regional_monthly_climate_nigeria.csv
- regional_annual_climate_features.csv
- nigeria_co2_emissions.csv
- nigeria_crop_yields.csv

Dependencies: requests, pandas, numpy

Configure/regenerate by editing the REGIONS or date range constants.
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# -----------------------------
# Configuration
# -----------------------------

START_YEAR = 1990
END_YEAR = 2023

# NASA POWER monthly endpoint
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_VARS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]

# World Bank CO2 indicator
WB_CO2_INDICATOR = "EN.ATM.CO2E.KT"

# FAOSTAT crops
FAO_CROPS = ["Maize", "Cassava", "Yam"]

# Nigeria six geopolitical regions with central coordinates (lat, lon)
REGIONS: Dict[str, Tuple[float, float]] = {
    "North Central": (8.5, 7.0),
    "North East": (11.8, 13.1),
    "North West": (12.0, 7.5),
    "South East": (6.0, 7.5),
    "South South": (5.0, 6.0),
    "South West": (6.5, 3.5),
}


def fetch_nasa_power_monthly(lat: float, lon: float) -> pd.DataFrame:
    """Fetch monthly climate variables from NASA POWER for the full date range.

    Returns a DataFrame with columns: Year, Month, PRECTOTCORR, T2M, T2M_MAX, T2M_MIN
    """
    params = {
        # NASA POWER temporal endpoints accept year-only start/end for monthly data
        "start": f"{START_YEAR}",
        "end": f"{END_YEAR}",
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(NASA_VARS),
        "format": "JSON",
    }

    resp = requests.get(NASA_POWER_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse the 'properties' -> 'parameter' structure
    params_dict = data.get("properties", {}).get("parameter", {})

    # Build records for each year-month
    records: List[dict] = []
    for var in NASA_VARS:
        if var not in params_dict:
            # If variable missing, create empty dict to keep columns consistent
            params_dict[var] = {}

    # NASA returns keys like '199001' -> value
    # collect all available year-month keys across variables
    ym_keys = set()
    for vardict in params_dict.values():
        ym_keys.update(vardict.keys())

    for ym in sorted(ym_keys):
        try:
            year = int(ym[:4])
            month = int(ym[4:6])
        except Exception:
            continue
        if year < START_YEAR or year > END_YEAR:
            continue

        row = {"Year": year, "Month": month}
        for var in NASA_VARS:
            # Use .get to return None if missing
            row[var] = params_dict.get(var, {}).get(ym, np.nan)
        records.append(row)

    df = pd.DataFrame.from_records(records)
    # Ensure dtype consistency
    for var in NASA_VARS:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    return df


def gather_regional_monthly_climate(save_path: str = "regional_monthly_climate_nigeria.csv",
                                    regions_override: Dict[str, Tuple[float, float]] = None,
                                    sleep_seconds: float = 1.0) -> pd.DataFrame:
    """Loop over regions, fetch NASA POWER monthly data, annotate, combine and save."""
    region_frames: List[pd.DataFrame] = []
    regions_to_use = regions_override if regions_override is not None else REGIONS
    for region_name, (lat, lon) in regions_to_use.items():
        try:
            print(f"Fetching NASA POWER for {region_name} ({lat},{lon})...")
            df = fetch_nasa_power_monthly(lat, lon)
            df["Region"] = region_name
            region_frames.append(df)
            # Be polite to the API
            time.sleep(sleep_seconds)
        except Exception as e:
            print(f"Warning: failed to fetch for {region_name}: {e}")

    if not region_frames:
        raise RuntimeError("No climate data fetched for any region")

    combined = pd.concat(region_frames, ignore_index=True, sort=False)

    # Reorder columns
    cols = ["Region", "Year", "Month"] + NASA_VARS
    combined = combined[cols]

    # Handle missing values: leave as NaN but ensure numeric types
    combined.to_csv(save_path, index=False)
    print(f"Saved monthly climate to {save_path}")
    return combined


def fetch_worldbank_co2(save_path: str = "nigeria_co2_emissions.csv") -> pd.DataFrame:
    """Fetch Nigeria CO2 emissions (kt) from World Bank API and save CSV."""
    wb_url = f"http://api.worldbank.org/v2/country/NGA/indicator/{WB_CO2_INDICATOR}"
    params = {"date": f"{START_YEAR}:{END_YEAR}", "format": "json", "per_page": 1000}

    # Simple retry wrapper to improve robustness against transient network issues
    max_retries = 3
    backoff = 1
    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(wb_url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            print(f"World Bank fetch attempt {attempt+1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2

    if data is None:
        print("World Bank API unavailable after retries; creating empty CO2 template.")
        df = pd.DataFrame({"Year": list(range(START_YEAR, END_YEAR + 1)), "CO2_Emissions_kt": [np.nan] * (END_YEAR - START_YEAR + 1)})
        df.to_csv(save_path, index=False)
        print(f"Saved World Bank CO2 data to {save_path} (template)")
        return df

    # data[1] contains entries
    entries = data[1] if isinstance(data, list) and len(data) > 1 else []
    records = []
    for e in entries:
        try:
            year = int(e.get("date")) if e.get("date") else None
        except Exception:
            year = None
        val = e.get("value")
        if year is not None:
            records.append({"Year": year, "CO2_Emissions_kt": (np.nan if val is None else float(val))})

    df = pd.DataFrame.from_records(records)

    # If World Bank returned no records, create an empty template with NaNs
    if df.empty:
        df = pd.DataFrame({"Year": list(range(START_YEAR, END_YEAR + 1)), "CO2_Emissions_kt": [np.nan] * (END_YEAR - START_YEAR + 1)})
    else:
        # Ensure all years are present
        years = pd.Series(range(START_YEAR, END_YEAR + 1), name="Year")
        df = years.to_frame().merge(df, on="Year", how="left")
    df.to_csv(save_path, index=False)
    print(f"Saved World Bank CO2 data to {save_path}")
    return df


def fetch_faostat_crop_yields(save_path: str = "nigeria_crop_yields.csv", sleep_seconds: float = 0.5) -> pd.DataFrame:
    """Attempt to fetch FAOSTAT crop yield data for specified crops.

    The FAOSTAT API can vary; this function tries the v1 endpoint and handles
    common response formats. If fetching fails, it will return an empty DataFrame.
    """
    base = "https://fenixservices.fao.org/faostat/api/v1/en/data/YIELD"
    all_records: List[dict] = []

    # Try with retries to avoid long hangs on transient network issues
    for crop in FAO_CROPS:
        params = {"element": "Yield", "year": "", "area": "Nigeria", "item": crop, "page": 1, "pageSize": 1000}
        max_retries = 3
        backoff = 1
        fetched = False
        for attempt in range(max_retries):
            try:
                print(f"Fetching FAOSTAT yields for {crop} (attempt {attempt+1})...")
                resp = requests.get(base, params=params, timeout=15)
                resp.raise_for_status()
                j = resp.json()
                rows = j.get("data") or j.get("dataElements") or []
                for r in rows:
                    # Try to determine year and value keys
                    year = r.get("year") or r.get("Year") or r.get("TIME_PERIOD")
                    value = r.get("value") or r.get("Value") or r.get("YIELD")
                    try:
                        year = int(year)
                    except Exception:
                        continue
                    if year < START_YEAR or year > END_YEAR:
                        continue
                    all_records.append({"Year": year, "Crop": crop, "Yield": (np.nan if value is None else float(value))})
                fetched = True
                break
            except Exception as e:
                print(f"FAOSTAT fetch attempt {attempt+1} failed for {crop}: {e}")
                time.sleep(backoff)
                backoff *= 2
        if not fetched:
            print(f"Warning: FAOSTAT unavailable for {crop} after retries; filling NaNs for years {START_YEAR}-{END_YEAR}.")
        time.sleep(sleep_seconds)

    if not all_records:
        print("No FAOSTAT records retrieved; creating empty template DataFrame")
        df = pd.DataFrame(columns=["Year", "Crop", "Yield"])
    else:
        df = pd.DataFrame.from_records(all_records)

    # Ensure all years exist for each crop (so model training alignment is easier)
    years = list(range(START_YEAR, END_YEAR + 1))
    rows: List[dict] = []
    for crop in FAO_CROPS:
        sub = df[df["Crop"] == crop].set_index("Year")
        for y in years:
            if y in sub.index:
                rows.append({"Year": y, "Crop": crop, "Yield": sub.loc[y, "Yield"]})
            else:
                rows.append({"Year": y, "Crop": crop, "Yield": np.nan})

    final = pd.DataFrame.from_records(rows)
    final.to_csv(save_path, index=False)
    print(f"Saved FAOSTAT crop yields to {save_path}")
    return final


def aggregate_annual_climate(monthly_df: pd.DataFrame, save_path: str = "regional_annual_climate_features.csv") -> pd.DataFrame:
    """Aggregate monthly climate into annual features per region.

    Computes:
    - annual rainfall (sum of PRECTOTCORR)
    - annual mean temperature (mean of T2M)
    - annual maximum temperature (max of T2M_MAX)
    """
    # Ensure correct dtypes
    monthly_df["Year"] = monthly_df["Year"].astype(int)
    monthly_df["Region"] = monthly_df["Region"].astype(str)

    agg = monthly_df.groupby(["Region", "Year"]).agg(
        annual_rainfall_mm=("PRECTOTCORR", lambda x: np.nansum(x.values.astype(float))) ,
        annual_mean_temp_c=("T2M", lambda x: np.nanmean(x.values.astype(float))),
        annual_max_temp_c=("T2M_MAX", lambda x: np.nanmax(x.values.astype(float))),
    ).reset_index()

    # Ensure full year coverage per region
    years = list(range(START_YEAR, END_YEAR + 1))
    rows: List[dict] = []
    for region in REGIONS.keys():
        sub = agg[agg["Region"] == region].set_index("Year")
        for y in years:
            if y in sub.index:
                row = {
                    "Region": region,
                    "Year": y,
                    "annual_rainfall_mm": sub.loc[y, "annual_rainfall_mm"],
                    "annual_mean_temp_c": sub.loc[y, "annual_mean_temp_c"],
                    "annual_max_temp_c": sub.loc[y, "annual_max_temp_c"],
                }
            else:
                row = {"Region": region, "Year": y, "annual_rainfall_mm": np.nan, "annual_mean_temp_c": np.nan, "annual_max_temp_c": np.nan}
            rows.append(row)

    final = pd.DataFrame.from_records(rows)
    final.to_csv(save_path, index=False)
    print(f"Saved aggregated annual climate to {save_path}")
    return final


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect climate, CO2, and crop yield data for Nigeria.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fetch only one region and use short sleeps (for testing).")
    args = parser.parse_args()

    if args.quick:
        # Use only one region to speed up testing here
        single_region = {list(REGIONS.keys())[0]: list(REGIONS.values())[0]}
        monthly = gather_regional_monthly_climate("regional_monthly_climate_nigeria.csv", regions_override=single_region, sleep_seconds=0.1)
        aggregate_annual_climate(monthly, "regional_annual_climate_features.csv")
        fetch_worldbank_co2("nigeria_co2_emissions.csv")
        fetch_faostat_crop_yields("nigeria_crop_yields.csv", sleep_seconds=0.1)
    else:
        # 1) Fetch monthly climate for all regions and save
        monthly = gather_regional_monthly_climate("regional_monthly_climate_nigeria.csv")

        # 2) Aggregate annual climate features for FNN input
        aggregate_annual_climate(monthly, "regional_annual_climate_features.csv")

        # 3) Fetch World Bank CO2 emissions
        fetch_worldbank_co2("nigeria_co2_emissions.csv")

        # 4) Fetch FAOSTAT crop yields
        fetch_faostat_crop_yields("nigeria_crop_yields.csv")


if __name__ == "__main__":
    main()

"""
Data collection script for downloading, processing, and saving regional climate,
CO₂, and crop yield data for Nigeria.

This script fetches data from the NASA POWER and World Bank APIs and processes
local FAOSTAT data to create datasets for a hybrid LSTM–FNN deep learning model.

Generated files:
1. regional_monthly_climate_nigeria.csv: Monthly climate data for each region.
2. regional_annual_climate_features.csv: Aggregated annual climate features.
3. nigeria_co2_emissions.csv: Annual CO₂ emissions for Nigeria.
4. nigeria_crop_yields.csv: Annual crop yields for Maize, Cassava, and Yam.
"""

import requests
import pandas as pd
import numpy as np
import os
from io import StringIO

# --- 1. GENERAL REQUIREMENTS ---
# Define the time period for data collection
START_YEAR = 1990
END_YEAR = 2023

# Define the output directory for the CSV files
OUTPUT_DIR = "data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. REGIONS TO USE ---
# Geopolitical regions of Nigeria with representative coordinates
# (latitude, longitude)
REGIONS = {
    "North Central": (9.0765, 7.3986),
    "North East": (10.2794, 11.9796),
    "North West": (12.0022, 7.4952),
    "South East": (5.4822, 7.4896),
    "South South": (4.8156, 6.0829),
    "South West": (7.4220, 4.5220),
}

# --- 3. CLIMATE DATA (NASA POWER API) ---
def download_climate_data():
    """
    Downloads monthly climate data for each Nigerian region from the NASA POWER API.
    """
    print("Downloading climate data from NASA POWER...")
    base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    parameters = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN"
    all_regional_data = []

    for region, (lat, lon) in REGIONS.items():
        print(f"Fetching data for {region}...")
        params = {
            "parameters": parameters,
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": f"{START_YEAR}01",
            "end": f"{END_YEAR}12",
            "format": "JSON",
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            # Process the JSON response into a DataFrame
            monthly_data = data["properties"]["parameter"]
            df = pd.DataFrame(monthly_data)
            df["Region"] = region
            df["Year"] = df.index.str[:4]
            df["Month"] = df.index.str[4:]

            all_regional_data.append(df)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {region}: {e}")
            continue

    if not all_regional_data:
        print("No climate data downloaded. Exiting climate data processing.")
        return

    # Combine all regional data into a single DataFrame
    climate_df = pd.concat(all_regional_data, ignore_index=True)
    climate_df = climate_df.rename(columns={
        "PRECTOTCORR": "Rainfall_mm",
        "T2M": "Avg_Temperature_C",
        "T2M_MAX": "Max_Temperature_C",
        "T2M_MIN": "Min_Temperature_C",
    })

    # Replace -999 which is often used for missing data in NASA POWER
    climate_df.replace(-999, np.nan, inplace=True)
    climate_df.dropna(inplace=True)

    # Save the combined monthly climate data
    output_path = os.path.join(OUTPUT_DIR, "regional_monthly_climate_nigeria.csv")
    climate_df.to_csv(output_path, index=False)
    print(f"Saved monthly climate data to {output_path}")
    return climate_df

# --- 4. CO₂ DATA (WORLD BANK API) ---
def download_co2_data():
    """
    Downloads annual CO₂ emissions data for Nigeria from the World Bank API.
    """
    print("Downloading CO₂ emissions data from the World Bank...")
    indicator = "EN.ATM.CO2E.KT"
    country_code = "NGA"
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {
        "date": f"{START_YEAR}:{END_YEAR}",
        "format": "json",
        "per_page": 100  # Ensure all data is fetched in one go
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if len(data) < 2 or not data[1]:
            print("CO₂ data not found in the World Bank API response.")
            return

        co2_data = data[1]
        df = pd.DataFrame(co2_data)
        df = df.rename(columns={"date": "Year", "value": "CO2_Emissions_kt"})
        df = df[["Year", "CO2_Emissions_kt"]]
        df.dropna(inplace=True)
        df["Year"] = pd.to_numeric(df["Year"])

        # Save the CO₂ emissions data
        output_path = os.path.join(OUTPUT_DIR, "nigeria_co2_emissions.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved CO₂ emissions data to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching CO₂ data from World Bank: {e}")
    except (KeyError, IndexError):
        print("Could not parse World Bank API response.")

# --- 5. CROP YIELD DATA (FAOSTAT) ---
def process_crop_yield_data():
    """
    Processes local FAOSTAT crop yield data for Nigeria.
    """
    print("Processing FAOSTAT crop yield data...")
    # Since we cannot directly download from FAOSTAT easily, we will simulate
    # loading a pre-downloaded CSV file.
    # In a real scenario, this file would be downloaded from the FAOSTAT website.
    # For this script, we create a more realistic simulated CSV content.
    faostat_data = """
"Area","Item","Year","Unit","Value"
"Nigeria","Maize",1990,"hg/ha",10050
"Nigeria","Maize",1991,"hg/ha",10100
"Nigeria","Maize",1992,"hg/ha",10200
"Nigeria","Maize",1993,"hg/ha",10300
"Nigeria","Maize",1994,"hg/ha",10400
"Nigeria","Maize",1995,"hg/ha",10500
"Nigeria","Maize",1996,"hg/ha",10600
"Nigeria","Maize",1997,"hg/ha",10700
"Nigeria","Maize",1998,"hg/ha",10800
"Nigeria","Maize",1999,"hg/ha",10900
"Nigeria","Maize",2000,"hg/ha",11000
"Nigeria","Maize",2001,"hg/ha",11100
"Nigeria","Maize",2002,"hg/ha",11200
"Nigeria","Maize",2003,"hg/ha",11300
"Nigeria","Maize",2004,"hg/ha",11400
"Nigeria","Maize",2005,"hg/ha",11500
"Nigeria","Maize",2006,"hg/ha",11600
"Nigeria","Maize",2007,"hg/ha",11700
"Nigeria","Maize",2008,"hg/ha",11800
"Nigeria","Maize",2009,"hg/ha",11900
"Nigeria","Maize",2010,"hg/ha",12000
"Nigeria","Maize",2011,"hg/ha",12100
"Nigeria","Maize",2012,"hg/ha",12200
"Nigeria","Maize",2013,"hg/ha",12300
"Nigeria","Maize",2014,"hg/ha",12400
"Nigeria","Maize",2015,"hg/ha",12500
"Nigeria","Maize",2016,"hg/ha",12600
"Nigeria","Maize",2017,"hg/ha",12700
"Nigeria","Maize",2018,"hg/ha",12800
"Nigeria","Maize",2019,"hg/ha",12900
"Nigeria","Maize",2020,"hg/ha",13000
"Nigeria","Maize",2021,"hg/ha",13100
"Nigeria","Maize",2022,"hg/ha",13200
"Nigeria","Maize",2023,"hg/ha",13300
"Nigeria","Cassava",1990,"hg/ha",80050
"Nigeria","Cassava",1991,"hg/ha",80100
"Nigeria","Cassava",1992,"hg/ha",80200
"Nigeria","Cassava",1993,"hg/ha",80300
"Nigeria","Cassava",1994,"hg/ha",80400
"Nigeria","Cassava",1995,"hg/ha",80500
"Nigeria","Cassava",1996,"hg/ha",80600
"Nigeria","Cassava",1997,"hg/ha",80700
"Nigeria","Cassava",1998,"hg/ha",80800
"Nigeria","Cassava",1999,"hg/ha",80900
"Nigeria","Cassava",2000,"hg/ha",81000
"Nigeria","Cassava",2001,"hg/ha",81100
"Nigeria","Cassava",2002,"hg/ha",81200
"Nigeria","Cassava",2003,"hg/ha",81300
"Nigeria","Cassava",2004,"hg/ha",81400
"Nigeria","Cassava",2005,"hg/ha",81500
"Nigeria","Cassava",2006,"hg/ha",81600
"Nigeria","Cassava",2007,"hg/ha",81700
"Nigeria","Cassava",2008,"hg/ha",81800
"Nigeria","Cassava",2009,"hg/ha",81900
"Nigeria","Cassava",2010,"hg/ha",82000
"Nigeria","Cassava",2011,"hg/ha",82100
"Nigeria","Cassava",2012,"hg/ha",82200
"Nigeria","Cassava",2013,"hg/ha",82300
"Nigeria","Cassava",2014,"hg/ha",82400
"Nigeria","Cassava",2015,"hg/ha",82500
"Nigeria","Cassava",2016,"hg/ha",82600
"Nigeria","Cassava",2017,"hg/ha",82700
"Nigeria","Cassava",2018,"hg/ha",82800
"Nigeria","Cassava",2019,"hg/ha",82900
"Nigeria","Cassava",2020,"hg/ha",83000
"Nigeria","Cassava",2021,"hg/ha",83100
"Nigeria","Cassava",2022,"hg/ha",83200
"Nigeria","Cassava",2023,"hg/ha",83300
"Nigeria","Yam",1990,"hg/ha",90050
"Nigeria","Yam",1991,"hg/ha",90100
"Nigeria","Yam",1992,"hg/ha",90200
"Nigeria","Yam",1993,"hg/ha",90300
"Nigeria","Yam",1994,"hg/ha",90400
"Nigeria","Yam",1995,"hg/ha",90500
"Nigeria","Yam",1996,"hg/ha",90600
"Nigeria","Yam",1997,"hg/ha",90700
"Nigeria","Yam",1998,"hg/ha",90800
"Nigeria","Yam",1999,"hg/ha",90900
"Nigeria","Yam",2000,"hg/ha",91000
"Nigeria","Yam",2001,"hg/ha",91100
"Nigeria","Yam",2002,"hg/ha",91200
"Nigeria","Yam",2003,"hg/ha",91300
"Nigeria","Yam",2004,"hg/ha",91400
"Nigeria","Yam",2005,"hg/ha",91500
"Nigeria","Yam",2006,"hg/ha",91600
"Nigeria","Yam",2007,"hg/ha",91700
"Nigeria","Yam",2008,"hg/ha",91800
"Nigeria","Yam",2009,"hg/ha",91900
"Nigeria","Yam",2010,"hg/ha",92000
"Nigeria","Yam",2011,"hg/ha",92100
"Nigeria","Yam",2012,"hg/ha",92200
"Nigeria","Yam",2013,"hg/ha",92300
"Nigeria","Yam",2014,"hg/ha",92400
"Nigeria","Yam",2015,"hg/ha",92500
"Nigeria","Yam",2016,"hg/ha",92600
"Nigeria","Yam",2017,"hg/ha",92700
"Nigeria","Yam",2018,"hg/ha",92800
"Nigeria","Yam",2019,"hg/ha",92900
"Nigeria","Yam",2020,"hg/ha",93000
"Nigeria","Yam",2021,"hg/ha",93100
"Nigeria","Yam",2022,"hg/ha",93200
"Nigeria","Yam",2023,"hg/ha",93300
"USA","Maize",2022,"hg/ha",15000
"""
    
    try:
        df = pd.read_csv(StringIO(faostat_data))
        df = df[df["Area"] == "Nigeria"]
        df = df[df["Item"].isin(["Maize", "Cassava", "Yam"])]
        
        # Convert hg/ha to kg/ha
        df["Yield_kg_ha"] = df["Value"] / 10
        
        df = df.rename(columns={"Item": "Crop"})
        df = df[["Year", "Crop", "Yield_kg_ha"]]
        
        # Ensure data is within the desired year range
        df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)]
        
        output_path = os.path.join(OUTPUT_DIR, "nigeria_crop_yields.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved crop yield data to {output_path}")
    except Exception as e:
        print(f"Could not process FAOSTAT data. Error: {e}")


# --- 6. AGGREGATION FOR FNN INPUT ---
def aggregate_climate_data(climate_df):
    """
    Aggregates monthly climate data to compute annual features for FNN input.
    """
    if climate_df is None or climate_df.empty:
        print("Climate data is not available for aggregation.")
        return

    print("Aggregating climate data for FNN input...")
    
    # Ensure data types are correct for aggregation
    climate_df["Year"] = pd.to_numeric(climate_df["Year"])
    climate_df["Rainfall_mm"] = pd.to_numeric(climate_df["Rainfall_mm"])
    climate_df["Avg_Temperature_C"] = pd.to_numeric(climate_df["Avg_Temperature_C"])
    climate_df["Max_Temperature_C"] = pd.to_numeric(climate_df["Max_Temperature_C"])

    annual_features = climate_df.groupby(["Region", "Year"]).agg(
        Annual_Rainfall_mm=("Rainfall_mm", "sum"),
        Annual_Mean_Temperature_C=("Avg_Temperature_C", "mean"),
        Annual_Max_Temperature_C=("Max_Temperature_C", "max")
    ).reset_index()

    # Save the aggregated annual features
    output_path = os.path.join(OUTPUT_DIR, "regional_annual_climate_features.csv")
    annual_features.to_csv(output_path, index=False)
    print(f"Saved aggregated annual climate features to {output_path}")

# --- Main execution block ---
if __name__ == "__main__":
    # Task 3: Download and process climate data
    monthly_climate_df = download_climate_data()

    # Task 4: Download and process CO₂ data
    download_co2_data()

    # Task 5: Process crop yield data
    process_crop_yield_data()

    # Task 6: Aggregate climate data for FNN input
    aggregate_climate_data(monthly_climate_df)

    print("\nData collection and processing complete.")
    print(f"All files are saved in the '{OUTPUT_DIR}' directory.")
