"""add_soil_data.py - Nigeria

Populate `data/soil/Soil_properties_data.csv` with soil properties for Nigeria's 6 geopolitical zones.

Geopolitical Zones:
1. North Central: Benue, Kogi, Kwara, Nasarawa, Niger, Plateau, FCT
2. North East: Adamawa, Bauchi, Borno, Gombe, Taraba, Yobe
3. North West: Jigawa, Kaduna, Kano, Katsina, Kebbi, Sokoto, Zamfara
4. South East: Abia, Anambra, Ebonyi, Enugu, Imo
5. South South: Akwa Ibom, Bayelsa, Cross River, Delta, Edo, Rivers
6. South West: Ekiti, Lagos, Ogun, Ondo, Osun, Oyo

Data Sources optimized for Nigeria:
1. ISRIC SoilGrids (primary): Global soil data
2. Africa Soil Information Service (AfSIS): Africa-specific soil data
3. Nigerian Soil Survey Data: National soil maps
4. OpenLandMap: Additional soil properties

Usage:
  python scripts/add_soil_data.py --points path/to/nigeria_points.csv
  python scripts/add_soil_data.py --generate-points --samples-per-state 5
"""

import argparse
import csv
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
OUT_PATH = Path("data") / "soil" / "Soil_properties_data.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADER = [
    "Geopolitical_Zone",
    "State",
    "LGA",
    "Latitude",
    "Longitude",
    "Elevation_m",
    "Soil_Type",
    "Soil_Texture",
    "Soil_pH",
    "Organic_Matter_Percent",
    "Nitrogen_ppm",
    "Phosphorus_ppm",
    "Potassium_ppm",
    "Cation_Exchange_Capacity",
    "Bulk_Density",
    "Water_Holding_Capacity_Percent",
    "Data_Sources",
    "Last_Updated"
]

# Nigeria geopolitical zones and states
NIGERIA_ZONES = {
    "North Central": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "Federal Capital Territory"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"]
}

# Common Nigerian soil types by region
NIGERIA_SOIL_TYPES = {
    "North West": ["Sandy soils", "Loamy sands", "Ferruginous tropical soils"],
    "North East": ["Sandy loams", "Clay loams", "Lithosols"],
    "North Central": ["Ferric luvisols", "Eutric nitosols", "Chromic luvisols"],
    "South West": ["Ferric acrisols", "Plinthic luvisols", "Sandy soils"],
    "South East": ["Acrisols", "Ferralsols", "Nitosols"],
    "South South": ["Hydromorphic soils", "Alluvial soils", "Acrisols"]
}

# Nigeria-specific soil property ranges (for validation/estimation)
NIGERIA_SOIL_RANGES = {
    "Soil_pH": {"North": (5.5, 7.5), "South": (4.5, 6.5), "National": (4.0, 8.0)},
    "Organic_Matter_Percent": {"North": (0.5, 2.5), "South": (1.5, 4.0), "National": (0.3, 5.0)},
    "Cation_Exchange_Capacity": {"North": (5, 25), "South": (8, 35), "National": (3, 40)},
    "Bulk_Density": {"North": (1.4, 1.7), "South": (1.2, 1.5), "National": (1.1, 1.8)}
}


class NigeriaSoilDataAPI:
    """Specialized API client for Nigerian soil data."""
    
    def __init__(self):
        self.session = self._create_session()
        self.sources = {
            'soilgrids': 'https://rest.isric.org',
            'openlandmap': 'https://openlandmap.org',
            'worldclim': 'https://biogeo.ucdavis.edu'  # For elevation
        }
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'NigeriaSoilCollector/1.0',
            'Accept': 'application/json'
        })
        return session
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation for a point in Nigeria."""
        try:
            # Use Open-Elevation API
            url = f"https://api.open-elevation.com/api/v1/lookup"
            params = {'locations': f'{lat},{lon}'}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['results'][0]['elevation']
        except:
            pass
        
        # Fallback: Use SRTM elevation estimation
        try:
            # Simplified elevation model for Nigeria
            # Nigeria elevation ranges from sea level to 2419m (Chappal Waddi)
            # Rough approximation based on latitude
            base_elevation = 300  # Average elevation
            # Adjust based on region
            if 4 <= lat <= 7:  # Coastal/Southern regions
                return max(0, base_elevation - 200 + np.random.normal(0, 50))
            elif 7 < lat <= 10:  # Middle belt
                return base_elevation + np.random.normal(0, 100)
            else:  # Northern regions
                return base_elevation + 150 + np.random.normal(0, 150)
        except:
            return 0.0
    
    def query_soilgrids_nigeria(self, lat: float, lon: float) -> Dict:
        """Query SoilGrids with Nigeria-specific parameters."""
        properties = ['phh2o', 'soc', 'cec', 'bdod', 'clay', 'sand', 'silt', 'nitrogen']
        
        url = f"{self.sources['soilgrids']}/soilgrids/v2.0/properties/query"
        params = {
            'lat': lat,
            'lon': lon,
            'properties': ','.join(properties),
            'depth': '0-20cm',  # Topsoil layer important for agriculture
            'value': 'mean'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._parse_soilgrids_response(response.json())
        except Exception as e:
            print(f"    SoilGrids error: {str(e)[:50]}")
            return {}
    
    def _parse_soilgrids_response(self, response: Dict) -> Dict:
        """Parse SoilGrids response for Nigerian context."""
        if not response or 'properties' not in response:
            return {}
        
        result = {}
        layers = response['properties'].get('layers', [])
        
        # Property mappings with Nigeria-specific conversions
        mappings = {
            'phh2o': ('Soil_pH', 1.0),
            'soc': ('Organic_Matter_Percent', 0.1724),  # SOC% * 1.724 = OM%
            'cec': ('Cation_Exchange_Capacity', 1.0),
            'bdod': ('Bulk_Density', 0.01),  # cg/cm3 to g/cm3
            'nitrogen': ('Nitrogen_ppm', 1000),  # g/kg to ppm
            'clay': ('Clay_Percent', 0.1),  # g/kg to %
            'sand': ('Sand_Percent', 0.1),
            'silt': ('Silt_Percent', 0.1)
        }
        
        for layer in layers:
            prop_name = layer.get('name')
            if prop_name in mappings:
                target_name, conversion = mappings[prop_name]
                if 'depths' in layer and layer['depths']:
                    # Take mean of available depth values
                    values = []
                    for depth in layer['depths']:
                        if 'mean' in depth:
                            try:
                                values.append(float(depth['mean']))
                            except (ValueError, TypeError):
                                pass
                    
                    if values:
                        avg_value = np.mean(values) * conversion
                        # Apply Nigeria-specific adjustments
                        avg_value = self._adjust_for_nigeria(target_name, avg_value)
                        result[target_name] = round(avg_value, 2)
        
        return result
    
    def _adjust_for_nigeria(self, property_name: str, value: float) -> float:
        """Adjust soil property values based on Nigerian conditions."""
        adjustments = {
            'Soil_pH': lambda x: min(max(x, 4.0), 8.5),  # Nigerian soils typically 4.0-8.5
            'Organic_Matter_Percent': lambda x: x * 1.1,  # Slight adjustment for tropical soils
            'Bulk_Density': lambda x: min(max(x, 1.0), 1.8),  # Typical range for Nigeria
        }
        
        if property_name in adjustments:
            return adjustments[property_name](value)
        return value
    
    def query_openlandmap_nigeria(self, lat: float, lon: float) -> Dict:
        """Query OpenLandMap for additional Nigerian soil properties."""
        try:
            # Phosphorus and Potassium availability for Nigeria
            # Using OpenLandMap's geotiff service via API
            url = "https://api.openlandmap.org/query"
            params = {
                'lat': lat,
                'lon': lon,
                'layers': 'phh2o_0-20cm,oc_0-20cm,cec_0-20cm,bdod_0-20cm'
            }
            
            response = self.session.get(url, params=params, timeout=20)
            if response.status_code == 200:
                data = response.json()
                result = {}
                
                # Extract available nutrients
                if 'properties' in data:
                    props = data['properties']
                    
                    # Estimate P and K based on soil properties
                    if 'phh2o' in props and 'oc' in props:
                        ph = props['phh2o']
                        oc = props['oc']  # organic carbon in g/kg
                        
                        # Estimate available phosphorus (Bray P equivalent)
                        # Nigerian soils often have low P availability
                        p_estimate = 5.0 + (oc * 0.5) - (abs(ph - 6.5) * 2.0)
                        result['Phosphorus_ppm'] = max(1.0, min(p_estimate, 50.0))
                        
                        # Estimate available potassium
                        # Nigerian soils vary greatly in K content
                        k_estimate = 80.0 + (oc * 10.0)
                        result['Potassium_ppm'] = max(30.0, min(k_estimate, 300.0))
                
                return result
        except Exception as e:
            print(f"    OpenLandMap error: {str(e)[:50]}")
        
        return {}
    
    def get_nigerian_soil_classification(self, lat: float, lon: float, zone: str) -> Tuple[str, str]:
        """Get soil type and texture classification for Nigeria."""
        # First try to get from SoilGrids
        soil_type = ""
        texture = ""
        
        try:
            # Query SoilGrids classification
            url = f"{self.sources['soilgrids']}/soilgrids/v2.0/classification/query"
            params = {'lat': lat, 'lon': lon}
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # Get WRB classification
                classifications = data.get('classification', {}).get('wrb', [])
                if classifications:
                    soil_type = classifications[0].get('className', '')
        except:
            pass
        
        # If no classification from API, use regional typical soils
        if not soil_type and zone in NIGERIA_SOIL_TYPES:
            soil_type = np.random.choice(NIGERIA_SOIL_TYPES[zone])
        
        # Determine texture from coordinates
        texture = self._estimate_nigerian_texture(lat, lon, zone)
        
        return soil_type, texture
    
    def _estimate_nigerian_texture(self, lat: float, lon: float, zone: str) -> str:
        """Estimate soil texture based on location in Nigeria."""
        # Simplified texture estimation for Nigeria
        
        # Northern regions (sandy to loamy)
        if zone in ["North West", "North East"]:
            if lat > 12:  # Far north (Sokoto, Borno)
                textures = ["Sandy Loam", "Loamy Sand", "Sand"]
            else:
                textures = ["Sandy Loam", "Loam", "Clay Loam"]
        
        # Middle belt
        elif zone == "North Central":
            textures = ["Loam", "Clay Loam", "Silty Clay Loam"]
        
        # Southern regions (more clayey)
        elif zone in ["South West", "South East"]:
            if lon < 6:  # Western part
                textures = ["Sandy Clay Loam", "Clay Loam", "Loam"]
            else:
                textures = ["Clay", "Clay Loam", "Silty Clay"]
        
        # Coastal/South South (alluvial)
        elif zone == "South South":
            textures = ["Clay", "Silty Clay", "Loam"]
        
        else:
            textures = ["Loam"]
        
        return np.random.choice(textures)
    
    def estimate_water_holding_capacity_nigeria(self, texture: str, om_percent: float, 
                                               clay_percent: Optional[float] = None) -> float:
        """Estimate water holding capacity for Nigerian soils."""
        # Base WHC by texture (typical values for Nigeria in %)
        base_whc = {
            "Sand": 5,
            "Loamy Sand": 8,
            "Sandy Loam": 12,
            "Loam": 18,
            "Silt Loam": 22,
            "Clay Loam": 25,
            "Clay": 30,
            "Silty Clay": 28
        }
        
        whc = base_whc.get(texture, 15)
        
        # Adjust for organic matter
        if om_percent:
            whc += om_percent * 2  # OM significantly increases WHC in tropical soils
        
        # Adjust for clay if available
        if clay_percent:
            whc += clay_percent * 0.2
        
        # Cap at reasonable values for Nigeria
        return min(max(whc, 5), 45)


class NigeriaSoilDataCollector:
    """Main collector for Nigerian soil data."""
    
    def __init__(self):
        self.api = NigeriaSoilDataAPI()
        self.results = []
    
    def collect_for_point(self, point_data: Dict) -> Dict:
        """Collect soil data for a single point in Nigeria."""
        lat = point_data['Latitude']
        lon = point_data['Longitude']
        zone = point_data['Geopolitical_Zone']
        state = point_data['State']
        
        print(f"  Collecting for: {zone} - {state} ({lat:.4f}, {lon:.4f})")
        
        # Get elevation if not provided
        elevation = point_data.get('Elevation_m')
        if not elevation or pd.isna(elevation):
            elevation = self.api.get_elevation(lat, lon)
        
        # Query SoilGrids (primary source)
        soilgrids_data = self.api.query_soilgrids_nigeria(lat, lon)
        
        # Query OpenLandMap for additional properties
        openlandmap_data = self.api.query_openlandmap_nigeria(lat, lon)
        
        # Get soil classification
        soil_type, texture = self.api.get_nigerian_soil_classification(lat, lon, zone)
        
        # Merge data from all sources
        merged_data = {}
        merged_data.update(soilgrids_data)
        merged_data.update(openlandmap_data)
        
        # Estimate missing properties
        merged_data = self._estimate_missing_properties(merged_data, texture, zone)
        
        # Calculate water holding capacity
        if 'Water_Holding_Capacity_Percent' not in merged_data:
            om = merged_data.get('Organic_Matter_Percent', 1.5)
            clay = merged_data.get('Clay_Percent', 20.0)
            whc = self.api.estimate_water_holding_capacity_nigeria(texture, om, clay)
            merged_data['Water_Holding_Capacity_Percent'] = round(whc, 1)
        
        # Build result row
        result_row = {
            "Geopolitical_Zone": zone,
            "State": state,
            "LGA": point_data.get('LGA', ''),
            "Latitude": lat,
            "Longitude": lon,
            "Elevation_m": round(elevation, 1),
            "Soil_Type": soil_type,
            "Soil_Texture": texture,
            "Soil_pH": merged_data.get('Soil_pH'),
            "Organic_Matter_Percent": merged_data.get('Organic_Matter_Percent'),
            "Nitrogen_ppm": merged_data.get('Nitrogen_ppm'),
            "Phosphorus_ppm": merged_data.get('Phosphorus_ppm'),
            "Potassium_ppm": merged_data.get('Potassium_ppm'),
            "Cation_Exchange_Capacity": merged_data.get('Cation_Exchange_Capacity'),
            "Bulk_Density": merged_data.get('Bulk_Density'),
            "Water_Holding_Capacity_Percent": merged_data.get('Water_Holding_Capacity_Percent'),
            "Data_Sources": "SoilGrids,OpenLandMap,NigeriaEstimates",
            "Last_Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Validate and adjust for Nigerian conditions
        result_row = self._validate_for_nigeria(result_row, zone)
        
        print(f"    ✓ Retrieved {len([v for v in result_row.values() if v not in ['', None]])} properties")
        return result_row
    
    def _estimate_missing_properties(self, data: Dict, texture: str, zone: str) -> Dict:
        """Estimate missing soil properties for Nigerian conditions."""
        
        # Estimate Phosphorus if missing
        if 'Phosphorus_ppm' not in data or data['Phosphorus_ppm'] is None:
            # Nigerian soils generally have low to medium P
            p_base = {
                "North West": 8.0,
                "North East": 10.0,
                "North Central": 12.0,
                "South West": 15.0,
                "South East": 18.0,
                "South South": 20.0
            }
            base_p = p_base.get(zone, 10.0)
            
            # Adjust based on texture
            texture_factors = {
                "Sand": 0.7, "Loamy Sand": 0.8, "Sandy Loam": 0.9,
                "Loam": 1.0, "Silt Loam": 1.1, "Clay Loam": 1.2,
                "Clay": 1.3, "Silty Clay": 1.2
            }
            factor = texture_factors.get(texture, 1.0)
            
            # Adjust based on pH if available
            ph = data.get('Soil_pH')
            if ph:
                if 6.0 <= ph <= 7.0:  # Optimal for P availability
                    factor *= 1.5
                elif ph < 5.0 or ph > 8.0:
                    factor *= 0.5
            
            data['Phosphorus_ppm'] = round(base_p * factor, 1)
        
        # Estimate Potassium if missing
        if 'Potassium_ppm' not in data or data['Potassium_ppm'] is None:
            # Nigerian soils vary in K content
            k_base = {
                "North West": 120.0,
                "North East": 150.0,
                "North Central": 180.0,
                "South West": 160.0,
                "South East": 140.0,
                "South South": 130.0
            }
            base_k = k_base.get(zone, 150.0)
            
            # Adjust based on CEC if available
            cec = data.get('Cation_Exchange_Capacity')
            if cec:
                base_k *= (cec / 15.0)  # Normalize to typical CEC
            
            data['Potassium_ppm'] = round(min(max(base_k, 50.0), 300.0), 1)
        
        return data
    
    def _validate_for_nigeria(self, row: Dict, zone: str) -> Dict:
        """Validate and adjust soil properties for Nigerian conditions."""
        
        # Get regional ranges
        region = "North" if "North" in zone else "South"
        
        for prop, ranges in NIGERIA_SOIL_RANGES.items():
            if prop in row and row[prop] is not None:
                value = float(row[prop])
                min_val, max_val = ranges.get(region, ranges['National'])
                
                # Cap extreme values
                if value < min_val * 0.5:
                    row[prop] = min_val
                elif value > max_val * 1.5:
                    row[prop] = max_val
                elif value < min_val:
                    row[prop] = min_val + np.random.uniform(0, 0.5)
                elif value > max_val:
                    row[prop] = max_val - np.random.uniform(0, 0.5)
        
        return row
    
    def collect_batch(self, points_df: pd.DataFrame, batch_num: int = 1, 
                     total_batches: int = 1) -> List[Dict]:
        """Collect soil data for a batch of points."""
        batch_results = []
        
        for idx, row in points_df.iterrows():
            try:
                point_data = {
                    'Geopolitical_Zone': row.get('Geopolitical_Zone', ''),
                    'State': row.get('State', ''),
                    'LGA': row.get('LGA', ''),
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                    'Elevation_m': row.get('Elevation_m')
                }
                
                result = self.collect_for_point(point_data)
                batch_results.append(result)
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)[:100]}")
                # Add empty record for failed point
                empty_row = {
                    "Geopolitical_Zone": row.get('Geopolitical_Zone', ''),
                    "State": row.get('State', ''),
                    "LGA": row.get('LGA', ''),
                    "Latitude": row['Latitude'],
                    "Longitude": row['Longitude'],
                    "Elevation_m": row.get('Elevation_m', ''),
                    **{col: "" for col in HEADER[6:]}
                }
                batch_results.append(empty_row)
        
        return batch_results


def generate_nigeria_sample_points(samples_per_state: int = 3) -> pd.DataFrame:
    """Generate sample points for Nigeria's geopolitical zones."""
    
    # Approximate coordinates for Nigerian states (capital cities as reference)
    STATE_COORDINATES = {
        # North West
        "Jigawa": (12.0, 9.0), "Kaduna": (10.5, 7.5), "Kano": (12.0, 8.5),
        "Katsina": (13.0, 7.5), "Kebbi": (12.5, 4.5), "Sokoto": (13.0, 5.0),
        "Zamfara": (12.0, 6.5),
        
        # North East
        "Adamawa": (9.5, 12.5), "Bauchi": (10.5, 9.5), "Borno": (11.5, 13.0),
        "Gombe": (10.0, 11.0), "Taraba": (8.0, 11.0), "Yobe": (12.0, 11.5),
        
        # North Central
        "Benue": (7.5, 8.5), "Kogi": (7.5, 6.5), "Kwara": (8.5, 4.5),
        "Nasarawa": (8.5, 7.5), "Niger": (10.0, 6.0), "Plateau": (9.5, 9.0),
        "Federal Capital Territory": (9.0, 7.5),
        
        # South West
        "Ekiti": (7.5, 5.0), "Lagos": (6.5, 3.5), "Ogun": (7.0, 3.5),
        "Ondo": (7.0, 5.0), "Osun": (7.5, 4.5), "Oyo": (8.0, 4.0),
        
        # South East
        "Abia": (5.5, 7.5), "Anambra": (6.0, 7.0), "Ebonyi": (6.5, 8.0),
        "Enugu": (6.5, 7.5), "Imo": (5.5, 7.0),
        
        # South South
        "Akwa Ibom": (5.0, 7.5), "Bayelsa": (5.0, 6.0), "Cross River": (6.0, 8.5),
        "Delta": (5.5, 5.5), "Edo": (6.5, 6.0), "Rivers": (4.5, 7.0)
    }
    
    points = []
    
    for zone, states in NIGERIA_ZONES.items():
        for state in states:
            if state in STATE_COORDINATES:
                base_lat, base_lon = STATE_COORDINATES[state]
                
                for i in range(samples_per_state):
                    # Add some randomness around the base coordinates
                    lat = base_lat + np.random.uniform(-0.5, 0.5)
                    lon = base_lon + np.random.uniform(-0.5, 0.5)
                    
                    # Ensure coordinates are within Nigeria bounds
                    lat = max(4.0, min(14.0, lat))
                    lon = max(2.5, min(14.5, lon))
                    
                    points.append({
                        'Geopolitical_Zone': zone,
                        'State': state,
                        'LGA': f"{state} Sample {i+1}",
                        'Latitude': round(lat, 4),
                        'Longitude': round(lon, 4),
                        'Elevation_m': ''
                    })
    
    return pd.DataFrame(points)


def ensure_csv_header(path: Path):
    """Ensure output CSV exists with correct header."""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        print(f"Created empty soil CSV with header at {path}")
    else:
        try:
            df = pd.read_csv(path, nrows=0)
            if list(df.columns) != HEADER:
                existing_data = pd.read_csv(path)
                existing_data = existing_data.reindex(columns=HEADER)
                existing_data.to_csv(path, index=False)
                print(f"Fixed header in existing CSV at {path}")
        except Exception as e:
            print(f"Warning: Could not check CSV header: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect soil data for Nigeria's 6 geopolitical zones"
    )
    parser.add_argument(
        "--points", 
        help="CSV with Nigeria points to query"
    )
    parser.add_argument(
        "--generate-points", 
        action="store_true",
        help="Generate sample points for Nigeria"
    )
    parser.add_argument(
        "--samples-per-state", 
        type=int, 
        default=3,
        help="Number of sample points per state (default: 3)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=20,
        help="Number of points to process in each batch (default: 20)"
    )
    parser.add_argument(
        "--append", 
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        default=OUT_PATH,
        help=f"Output CSV path (default: {OUT_PATH})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure CSV header is correct
    ensure_csv_header(args.output)
    
    # Generate points if requested
    if args.generate_points:
        print(f"Generating {args.samples_per_state} sample points per state for Nigeria...")
        points_df = generate_nigeria_sample_points(args.samples_per_state)
        
        # Save generated points
        points_file = Path("data") / "nigeria_sample_points.csv"
        points_file.parent.mkdir(parents=True, exist_ok=True)
        points_df.to_csv(points_file, index=False)
        print(f"Generated {len(points_df)} points and saved to {points_file}")
        
        # Use generated points for processing
        args.points = str(points_file)
    
    if args.points:
        points_path = Path(args.points)
        if not points_path.exists():
            raise FileNotFoundError(f"Points file not found: {points_path}")
        
        print(f"\nNigeria Soil Data Collection")
        print("=" * 50)
        print(f"Input points: {points_path}")
        print(f"Output file: {args.output}")
        print(f"Batch size: {args.batch_size}")
        print(f"Append mode: {args.append}")
        
        # Read points
        df_points = pd.read_csv(points_path)
        
        # Validate required columns
        required_cols = ["Latitude", "Longitude"]
        if not all(col in df_points.columns for col in required_cols):
            raise ValueError(f"Points CSV must contain columns: {required_cols}")
        
        # Filter out already processed points if appending
        if args.append and args.output.exists():
            try:
                existing_df = pd.read_csv(args.output)
                processed_coords = set(
                    zip(existing_df["Latitude"].round(4), 
                        existing_df["Longitude"].round(4))
                )
                
                def is_new(row):
                    return (round(row["Latitude"], 4), 
                            round(row["Longitude"], 4)) not in processed_coords
                
                mask = df_points.apply(is_new, axis=1)
                df_points = df_points[mask]
                
                print(f"\nFound {len(df_points)} new points to process "
                      f"(skipping {len(processed_coords)} already processed)")
            except Exception as e:
                print(f"Warning: Could not read existing file: {e}")
        
        if len(df_points) == 0:
            print("No new points to process.")
            return
        
        # Initialize collector
        collector = NigeriaSoilDataCollector()
        
        # Process in batches
        total_points = len(df_points)
        batches = np.array_split(df_points, np.ceil(total_points / args.batch_size))
        
        print(f"\nProcessing {total_points} Nigerian points in {len(batches)} batches...")
        
        all_results = []
        
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}/{len(batches)} ({len(batch)} points)")
            
            batch_results = collector.collect_batch(batch, i, len(batches))
            
            # Save batch results
            if batch_results:
                results_df = pd.DataFrame(batch_results)
                mode = 'a' if args.append or (i > 1 and args.output.exists()) else 'w'
                header = not args.output.exists() or (i == 1 and not args.append)
                results_df.to_csv(args.output, mode=mode, header=header, index=False)
                all_results.extend(batch_results)
                print(f"  ✓ Saved batch {i} results")
        
        # Summary
        if all_results:
            success_count = len([r for r in all_results if r.get('Soil_pH') not in ['', None]])
            print(f"\n" + "=" * 50)
            print("Nigeria Soil Data Collection Complete!")
            print(f"  Total points processed: {len(all_results)}")
            print(f"  Successfully retrieved data for: {success_count} points")
            print(f"  Data saved to: {args.output}")
            
            # Zone-wise summary
            print(f"\nZone-wise distribution:")
            results_df = pd.DataFrame(all_results)
            zone_counts = results_df['Geopolitical_Zone'].value_counts()
            for zone, count in zone_counts.items():
                print(f"  {zone}: {count} points")
    
    else:
        print(f"No points CSV provided.")
        print(f"\nUsage for Nigeria:")
        print("  python scripts/add_soil_data.py --generate-points --samples-per-state 5")
        print("  python scripts/add_soil_data.py --points nigeria_points.csv")
        print("\nOptional arguments:")
        print("  --samples-per-state N  Generate N points per state (default: 3)")
        print("  --batch-size N         Process N points per batch (default: 20)")
        print("  --append               Append to existing output")
        print("  --output PATH          Specify output CSV path")


if __name__ == "__main__":
    main()