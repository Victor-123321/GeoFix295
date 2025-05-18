# GeoFix295: POI295 Validation Correction Tool
# Early development for HERE Technologies GuadalaHacks 2025
# Implements data loading and POI geometry computation

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import logging
import argparse

# Set up logging
logging.basicConfig(filename='poi295_corrections.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def load_datasets(poi_folder, streets_nav_folder, streets_naming_folder, mode='test'):
    """Load and merge POI, Streets Nav, and Naming datasets based on mode."""
    file_limit = 1 if mode == 'test' else None
    print(f"Running in {mode} mode: {'1 file per dataset' if mode == 'test' else 'all files'}")
    
    # Load POIs
    poi_files = [os.path.join(poi_folder, f) for f in os.listdir(poi_folder) if f.endswith('.csv')]
    print(f"Loading {len(poi_files[:file_limit])} POI files...")
    poi_dfs = [pd.read_csv(f) for f in poi_files[:file_limit]]
    poi_df = pd.concat(poi_dfs, ignore_index=True)
    logging.info(f"Loaded {len(poi_dfs)} POI files with {len(poi_df)} records")
    del poi_dfs
    
    # Load Streets Nav
    nav_files = [os.path.join(streets_nav_folder, f) for f in os.listdir(streets_nav_folder) if f.endswith('.geojson')]
    print(f"Loading {len(nav_files[:file_limit])} Streets Nav files...")
    nav_gdfs = [gpd.read_file(f) for f in nav_files[:file_limit]]
    streets_nav = gpd.GeoDataFrame(pd.concat([gdf for gdf in nav_gdfs], ignore_index=True), crs="EPSG:4326")
    logging.info(f"Loaded {len(nav_gdfs)} Streets Nav files with {len(streets_nav)} features")
    del nav_gdfs
    
    # Load Streets Naming & Addressing
    naming_files = [os.path.join(streets_naming_folder, f) for f in os.listdir(streets_naming_folder) if f.endswith('.geojson')]
    print(f"Loading {len(naming_files[:file_limit])} Streets Naming files...")
    naming_gdfs = [gpd.read_file(f) for f in naming_files[:file_limit]]
    streets_naming = gpd.GeoDataFrame(pd.concat([gdf for gdf in naming_gdfs], ignore_index=True), crs="EPSG:4326")
    logging.info(f"Loaded {len(naming_gdfs)} Streets Naming files with {len(streets_naming)} features")
    del naming_gdfs
    
    return poi_df, streets_nav, streets_naming

def compute_poi_geometry(poi_df, streets_nav):
    """Compute POI geometries based on LINK_ID, PERCFRREF, and POI_ST_SD."""
    poi_gdf = poi_df.copy()
    geometries = []
    for idx, row in poi_gdf.iterrows():
        if idx % 1000 == 0:
            print(f"Processing POI {idx}/{len(poi_gdf)}")
        link_id = row['LINK_ID']
        percfrref = row['PERCFRREF'] / 100.0  # Convert to fraction
        poi_st_sd = row['POI_ST_SD']
        
        # Find matching link
        link = streets_nav[streets_nav['link_id'] == link_id]
        if link.empty:
            logging.warning(f"No link found for LINK_ID {link_id} for POI {row['POI_ID']}")
            geometries.append(None)
            continue
        
        # Get link geometry
        line = link.iloc[0].geometry
        # Interpolate point along link
        point = line.interpolate(percfrref, normalized=True)
        
        # Offset point based on POI_ST_SD (left/right)
        # For simplicity, assume small offset (e.g., 5 meters) perpendicular to link
        if poi_st_sd == 'L':
            # Offset to left (negative direction)
            angle = 90  # Perpendicular
        elif poi_st_sd == 'R':
            # Offset to right (positive direction)
            angle = -90
        else:
            angle = 0  # No offset if side unknown
        if angle != 0:
            # Calculate offset (mock; use real projection for production)
            offset = 5 / 111000  # Approx 5m in degrees (crude estimate)
            dx = offset * (angle / 90)
            point = Point(point.x + dx, point.y)
        
        geometries.append(point)
    
    poi_gdf['geometry'] = geometries
    return gpd.GeoDataFrame(poi_gdf, crs="EPSG:4326")

def classify_scenario(poi, poi_point, link1, link2, streets_nav, streets_naming):
    """Classify POI295 violation into one of four scenarios."""
    poi_id = poi['POI_ID']
    poi_name = poi['POI_NAME']
    link_id = poi['LINK_ID']
    poi_st_sd = poi['POI_ST_SD']
    
    # Scenario 1: No POI in reality
    if not check_poi_existence(poi_point, poi_name):
        return 1, "Mark for deletion"
    
    # Scenario 2: Incorrect POI location
    # Check if POI is on correct side by analyzing nearby links
    nearby_links = streets_nav[streets_nav.geometry.distance(poi_point) < 10 / 111000]  # ~10m
    correct_link = None
    for _, nearby in nearby_links.iterrows():
        if nearby['link_id'] != link_id and nearby.geometry.distance(poi_point) < 5 / 111000:
            correct_link = nearby
            break
    if correct_link is not None:
        return 2, f"Update LINK_ID to {correct_link['link_id']}"
    
    # Scenario 3: Incorrect MULTIDIGIT attribution
    naming1 = streets_naming[streets_naming['link_id'] == link1['link_id']]
    naming2 = streets_naming[streets_naming['link_id'] == link2['link_id']]
    if naming1.empty or naming2.empty:
        logging.warning(f"Missing naming data for link1 {link1['link_id']} or link2 {link2['link_id']}")
        return 3, "Set MULTIDIGIT = 'N' for both links"  # Fallback if no naming data
    same_name = naming1.iloc[0]['ST_NAME'] == naming2.iloc[0]['ST_NAME']
    no_vegetation = not check_vegetation(link_id, poi_point)
    distance_ok = link1.geometry.distance(link2.geometry) < 30 / 111000  # ~30m threshold
    if not (same_name and no_vegetation and distance_ok):
        return 3, "Set MULTIDIGIT = 'N' for both links"
    
    # Scenario 4: Legitimate Exception
    return 4, "Mark as Legitimate Exception"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process POI295 validations')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='Mode: test (1 file per dataset) or full (all files)')
    args = parser.parse_args()
    
    # File paths
    poi_folder = 'POIs'
    streets_nav_folder = 'STREETS_NAV'
    streets_naming_folder = 'STREETS_NAMING_ADDRESSING'
    
    # Load datasets
    print("Loading datasets...")
    poi_df, streets_nav, streets_naming = load_datasets(poi_folder, streets_nav_folder, streets_naming_folder, mode=args.mode)
    
    # Compute POI geometries
    print(f"Computing POI geometries for {len(poi_df)} POIs...")
    poi_gdf = compute_poi_geometry(poi_df, streets_nav)
    
    # Placeholder for future violation detection and correction
    print("Geometry computation complete. Violation detection and corrections coming soon!")
    logging.info(f"Processed {len(poi_gdf)} POIs with geometries")
    
    # Save temporary output for debugging
    poi_gdf.drop(columns=['geometry']).to_csv('temp_pois.csv', index=False)
    print("Saved temporary output to temp_pois.csv")

if __name__ == '__main__':
    main()