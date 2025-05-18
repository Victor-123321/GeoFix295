import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os
import uuid
import logging
import argparse
import re

# Set up logging
logging.basicConfig(filename='poi295_corrections.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Mock function for satellite imagery analysis (placeholder)
def check_poi_existence(coordinates, poi_name):
    """Mock function to check POI existence via satellite imagery."""
    # TODO: Implement HERE Raster Tile API call with API key
    # For now, return True (assume POI exists) for testing
    logging.info(f"Mock imagery check for {poi_name} at {coordinates}: Assumed to exist")
    return True

def check_vegetation(link_id, coordinates):
    """Mock function to check vegetation for MULTIDIGIT validation."""
    # TODO: Implement imagery analysis for vegetation detection
    # For now, return False (assume no heavy vegetation)
    logging.info(f"Mock vegetation check for link {link_id} at {coordinates}: No vegetation")
    return False

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

def find_multidigit_pairs(streets_nav, streets_naming):
    """Identify pairs of MULTIDIGIT links with same ST_NAME."""
    multidigit_links = streets_nav[streets_nav['MULTIDIGIT'] == 'Y']
    pairs = []
    max_links = 1000  # Temporary limit for testing
    for idx1, link1 in multidigit_links.head(max_links).iterrows():
        link_id1 = link1['link_id']
        geom1 = link1.geometry
        # Find matching street name
        naming1 = streets_naming[streets_naming['link_id'] == link_id1]
        if naming1.empty:
            continue
        st_name1 = naming1.iloc[0]['ST_NAME']
        
        # Find potential pair
        for idx2, link2 in multidigit_links.iloc[idx1+1:].head(max_links).iterrows():
            naming2 = streets_naming[streets_naming['link_id'] == link2['link_id']]
            if naming2.empty:
                continue
            st_name2 = naming2.iloc[0]['ST_NAME']
            
            # Check if same street and close proximity
            if st_name1 == st_name2 and geom1.distance(link2.geometry) < 50 / 111000:  # ~50m
                pairs.append((link_id1, link2['link_id'], geom1, link2.geometry))
    
    logging.info(f"Found {len(pairs)} MULTIDIGIT pairs")
    return pairs

def check_inside_multidigit(poi_point, link1_geom, link2_geom):
    """Check if POI lies between two MULTIDIGIT links (inside)."""
    if not (poi_point and link1_geom.is_valid and link2_geom.is_valid):
        return False
    try:
        # Combine coordinates of both links to form a polygon
        coords = list(link1_geom.coords) + list(link2_geom.coords)[::-1]  # Reverse second link for closed polygon
        polygon = Polygon(coords)
        return polygon.contains(poi_point)
    except:
        return False

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

def apply_corrections(poi_gdf, streets_nav, violations):
    """Apply corrections based on scenario classification."""
    validations = []
    for _, violation in violations.iterrows():
        poi_id = violation['POI_ID']
        scenario = violation['scenario']
        action = violation['action']
        poi_idx = poi_gdf[poi_gdf['POI_ID'] == poi_id].index[0]
        link_id = poi_gdf.loc[poi_idx, 'LINK_ID']
        
        if scenario == 1:
            # Delete POI
            poi_gdf = poi_gdf.drop(poi_idx)
            validations.append({'POI_ID': poi_id, 'Action': 'Deleted'})
        elif scenario == 2:
            # Update LINK_ID
            match = re.search(r'Update LINK_ID to (\d+)', action)
            if match:
                new_link_id = match.group(1)
                poi_gdf.loc[poi_idx, 'LINK_ID'] = int(new_link_id)
            else:
                logging.warning(f"Could not parse LINK_ID from action: {action}")
                validations.append({'POI_ID': poi_id, 'Action': 'Skipped: Invalid LINK_ID'})
                continue
            validations.append({'POI_ID': poi_id, 'Action': action})
        elif scenario == 3:
            # Update MULTIDIGIT
            link1_idx = streets_nav[streets_nav['link_id'] == violation['link_id1']].index[0]
            link2_idx = streets_nav[streets_nav['link_id'] == violation['link_id2']].index[0]
            streets_nav.loc[link1_idx, 'MULTIDIGIT'] = 'N'
            streets_nav.loc[link2_idx, 'MULTIDIGIT'] = 'N'
            validations.append({'POI_ID': poi_id, 'Action': action})
        elif scenario == 4:
            # Mark as exception
            validations.append({'POI_ID': poi_id, 'Action': 'Legitimate Exception'})
        
        logging.info(f"POI {poi_id}: Scenario {scenario} - {action}")
    
    return poi_gdf, streets_nav, pd.DataFrame(validations)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process POI295 validations')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='Mode: test (1 file per dataset) or full (all files)')
    args = parser.parse_args()
    
    # File paths (update as needed)
    poi_folder = 'POIs'
    streets_nav_folder = 'STREETS_NAV'
    streets_naming_folder = 'STREETS_NAMING_ADDRESSING'
    
    # Load datasets
    print("Loading datasets...")
    poi_df, streets_nav, streets_naming = load_datasets(poi_folder, streets_nav_folder, streets_naming_folder, mode=args.mode)
    
    # Compute POI geometries
    print(f"Computing POI geometries for {len(poi_df)} POIs...")
    poi_gdf = compute_poi_geometry(poi_df, streets_nav)
    
    # Find MULTIDIGIT pairs
    print("Finding MULTIDIGIT pairs...")
    multidigit_pairs = find_multidigit_pairs(streets_nav, streets_naming)
    
    # Detect POI295 violations
    print(f"Detecting violations among {len(multidigit_pairs)} MULTIDIGIT pairs...")
    violations = []
    for i, (link_id1, link_id2, geom1, geom2) in enumerate(multidigit_pairs):
        pois = poi_gdf[poi_gdf['LINK_ID'].isin([link_id1, link_id2])]
        for _, poi in pois.iterrows():
            poi_point = poi.geometry
            if poi_point and check_inside_multidigit(poi_point, geom1, geom2):
                scenario, action = classify_scenario(
                    poi,
                    poi_point,
                    streets_nav[streets_nav['link_id'] == link_id1].iloc[0],
                    streets_nav[streets_nav['link_id'] == link_id2].iloc[0],
                    streets_nav,
                    streets_naming
                )
                violations.append({
                    'POI_ID': poi['POI_ID'],
                    'link_id1': link_id1,
                    'link_id2': link_id2,
                    'scenario': scenario,
                    'action': action
                })
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(multidigit_pairs)} MULTIDIGIT pairs")
    
    # Apply corrections
    print("Applying corrections...")
    updated_poi_gdf, updated_streets_nav, validations_df = apply_corrections(poi_gdf, streets_nav, pd.DataFrame(violations))
    
    # Save outputs
    print("Saving outputs...")
    updated_poi_gdf.drop(columns=['geometry']).to_csv('updated_pois.csv', index=False)
    updated_streets_nav.to_file('updated_streets_nav.geojson', driver='GeoJSON')
    validations_df.to_csv('validations.csv', index=False)
    
    print(f"Processed {len(violations)} POI295 violations. Check updated_pois.csv, updated_streets_nav.geojson, and validations.csv")

if __name__ == '__main__':
    main()