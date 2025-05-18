import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os
import logging
import argparse
import re
from rtree import index
import folium
from folium.plugins import HeatMap
import sys
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='poi295_corrections.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Mock functions
def check_poi_existence(coordinates, poi_name):
    """Mock function to check POI existence via satellite imagery."""
    logging.info(f"Mock imagery check for {poi_name} at {coordinates}: Assumed to exist")
    return True

def check_vegetation(link_id, coordinates):
    """Mock function to check vegetation for MULTIDIGIT validation."""
    logging.info(f"Mock vegetation check for link {link_id} at {coordinates}: No vegetation")
    return False

def load_datasets(poi_folder, streets_nav_folder, streets_naming_folder, mode='test', tile_id=None):
    """Load and merge POI, Streets Nav, and Naming datasets based on mode and optional tile_id."""
    logging.info(f"Starting data loading in {mode} mode, tile_id={tile_id}")
    file_limit = 1 if mode == 'test' else None
    print(f"Running in {mode} mode: {'1 file per dataset' if mode == 'test' else 'all files'}")
    
    # Filter files by tile_id if provided
    if tile_id:
        poi_files = [os.path.join(poi_folder, f) for f in os.listdir(poi_folder) 
                     if f.endswith('.csv') and tile_id in f]
        nav_files = [os.path.join(streets_nav_folder, f) for f in os.listdir(streets_nav_folder) 
                     if f.endswith('.geojson') and tile_id in f]
        naming_files = [os.path.join(streets_naming_folder, f) for f in os.listdir(streets_naming_folder) 
                        if f.endswith('.geojson') and tile_id in f]
    else:
        poi_files = [os.path.join(poi_folder, f) for f in os.listdir(poi_folder) if f.endswith('.csv')]
        nav_files = [os.path.join(streets_nav_folder, f) for f in os.listdir(streets_nav_folder) if f.endswith('.geojson')]
        naming_files = [os.path.join(streets_naming_folder, f) for f in os.listdir(streets_naming_folder) if f.endswith('.geojson')]
    
    # Apply file limit for test mode
    poi_files = poi_files[:file_limit]
    nav_files = nav_files[:file_limit]
    naming_files = naming_files[:file_limit]
    
    # Load POI files
    print(f"Loading {len(poi_files)} POI files...")
    try:
        poi_dfs = [pd.read_csv(f) for f in poi_files]
        poi_df = pd.concat(poi_dfs, ignore_index=True) if poi_dfs else pd.DataFrame()
        if not poi_df.empty:
            poi_df['POI_ID'] = poi_df['POI_ID'].astype(str)  # Standardize to string
        logging.info(f"Loaded {len(poi_dfs)} POI files with {len(poi_df)} records")
        del poi_dfs
    except Exception as e:
        logging.error(f"Failed to load POI files: {e}")
        raise
    
    # Load Streets Nav files
    print(f"Loading {len(nav_files)} Streets Nav files...")
    try:
        nav_gdfs = [gpd.read_file(f) for f in nav_files]
        streets_nav = gpd.GeoDataFrame(pd.concat([gdf for gdf in nav_gdfs], ignore_index=True), crs="EPSG:4326") if nav_gdfs else gpd.GeoDataFrame()
        if not streets_nav.empty:
            logging.info(f"Loaded {len(nav_gdfs)} Streets Nav files with {len(streets_nav)} features")
            logging.info(f"MULTIDIGIT = 'Y' links: {len(streets_nav[streets_nav['MULTIDIGIT'] == 'Y'])}")
        del nav_gdfs
    except Exception as e:
        logging.error(f"Failed to load Streets Nav files: {e}")
        raise
    
    # Load Streets Naming files
    print(f"Loading {len(naming_files)} Streets Naming files...")
    try:
        naming_gdfs = [gpd.read_file(f) for f in naming_files]
        streets_naming = gpd.GeoDataFrame(pd.concat([gdf for gdf in naming_gdfs], ignore_index=True), crs="EPSG:4326") if naming_gdfs else gpd.GeoDataFrame()
        if not streets_naming.empty:
            logging.info(f"Loaded {len(naming_gdfs)} Streets Naming files with {len(streets_naming)} features")
        del naming_gdfs
    except Exception as e:
        logging.error(f"Failed to load Streets Naming files: {e}")
        raise
    
    logging.info("Completed data loading")
    return poi_df, streets_nav, streets_naming

def compute_poi_geometry(poi_df, streets_nav):
    """Compute POI geometries based on LINK_ID, PERCFRREF, and POI_ST_SD."""
    logging.info(f"Starting geometry computation for {len(poi_df)} POIs")
    poi_gdf = poi_df.copy()
    geometries = []
    null_count = 0
    for idx, row in poi_gdf.iterrows():
        if idx % 1000 == 0:
            logging.info(f"Computed geometry for POI {idx}/{len(poi_gdf)}")
        link_id = row['LINK_ID']
        percfrref = row['PERCFRREF'] / 100.0
        poi_st_sd = row['POI_ST_SD']
        
        link = streets_nav[streets_nav['link_id'] == link_id]
        if link.empty:
            logging.warning(f"No link found for LINK_ID {link_id} for POI {row['POI_ID']}")
            geometries.append(None)
            null_count += 1
            continue
        
        line = link.iloc[0].geometry
        if not line.is_valid:
            logging.warning(f"Invalid geometry for LINK_ID {link_id} for POI {row['POI_ID']}")
            geometries.append(None)
            null_count += 1
            continue
        
        point = line.interpolate(percfrref, normalized=True)
        if poi_st_sd == 'L':
            angle = 90
        elif poi_st_sd == 'R':
            angle = -90
        else:
            angle = 0
        if angle != 0:
            offset = 5 / 111000  # 5m offset
            dx = offset * (angle / 90)
            point = Point(point.x - dx * 0.5, point.y)
        
        geometries.append(point)
    
    poi_gdf['geometry'] = geometries
    poi_gdf = gpd.GeoDataFrame(poi_gdf, crs="EPSG:4326")
    logging.info(f"Completed geometry computation for {len(poi_gdf)} POIs")
    logging.info(f"Null geometries: {null_count} ({null_count/len(poi_gdf)*100:.2f}%)")
    return poi_gdf

def find_multidigit_pairs(streets_nav, streets_naming):
    """Identify pairs of MULTIDIGIT links with same ST_NAME using rtree."""
    logging.info("Starting MULTIDIGIT pair detection")
    multidigit_links = streets_nav[streets_nav['MULTIDIGIT'] == 'Y']
    
    if multidigit_links.empty:
        logging.warning("No MULTIDIGIT links found in dataset")
        return []
    
    idx = index.Index()
    df_indices = multidigit_links.index.tolist()
    for i, df_idx in enumerate(df_indices):
        link = multidigit_links.loc[df_idx]
        idx.insert(df_idx, link.geometry.bounds)
    
    pairs = []
    for idx1, link1 in multidigit_links.iterrows():
        link_id1 = link1['link_id']
        geom1 = link1.geometry
        naming1 = streets_naming[streets_naming['link_id'] == link_id1]
        if naming1.empty:
            logging.warning(f"No ST_NAME for link_id {link_id1}")
            continue
        st_name1 = naming1.iloc[0]['ST_NAME']
        
        for idx2 in idx.intersection(geom1.bounds):
            if idx2 <= idx1:
                continue
            if idx2 not in df_indices:
                logging.warning(f"Invalid rtree index {idx2} for multidigit_links")
                continue
            link2 = multidigit_links.loc[idx2]
            naming2 = streets_naming[streets_naming['link_id'] == link2['link_id']]
            if naming2.empty or naming1.iloc[0]['ST_NAME'] != naming2.iloc[0]['ST_NAME']:
                continue
            if geom1.distance(link2.geometry) < 50 / 111000:  # 50m
                pairs.append((link_id1, link2['link_id'], geom1, link2.geometry))
                logging.info(f"Pair: {link_id1}, {link2['link_id']}, Distance: {geom1.distance(link2.geometry)*111000}m")
    
    logging.info(f"Found {len(pairs)} MULTIDIGIT pairs")
    return pairs

def check_inside_multidigit(poi_point, link1_geom, link2_geom, link_id1, link_id2):
    """Check if POI lies between two MULTIDIGIT links using a buffered polygon."""
    if not (poi_point and link1_geom.is_valid and link2_geom.is_valid):
        logging.warning(f"Invalid input for check_inside_multidigit: link_id1={link_id1}, link_id2={link_id2}")
        return False
    try:
        if link1_geom.almost_equals(link2_geom, decimal=6):
            logging.warning(f"Links {link_id1}, {link_id2} are nearly identical")
            return False
        if link1_geom.distance(link2_geom) < 1 / 111000:  # ~1m
            logging.warning(f"Links {link_id1}, {link_id2} too close: {link1_geom.distance(link2_geom)*111000}m")
            return False

        buffer_dist = 10 / 111000  # 10m buffer
        union_geom = link1_geom.buffer(buffer_dist).union(link2_geom.buffer(buffer_dist))
        if union_geom.geom_type != 'Polygon':
            union_geom = union_geom.convex_hull
        if not union_geom.is_valid:
            logging.warning(f"Invalid polygon for links {link_id1}, {link_id2}: {union_geom}")
            return False

        result = union_geom.contains(poi_point)
        if result:
            logging.info(f"POI at {poi_point} is between links {link_id1}, {link_id2}")
        return result
    except Exception as e:
        logging.warning(f"Error in check_inside_multidigit for links {link_id1}, {link_id2}: {e}")
        return False

def classify_scenario(poi, poi_point, link1, link2, streets_nav, streets_naming):
    """Classify POI295 violation into one of four scenarios."""
    poi_id = str(poi['POI_ID'])
    poi_name = poi['POI_NAME']
    link_id = poi['LINK_ID']
    logging.info(f"Classifying scenario for POI {poi_id}")
    
    if not check_poi_existence(poi_point, poi_name):
        return 1, "Deleted"
    
    nearby_links = streets_nav[streets_nav.geometry.distance(poi_point) < 10 / 111000]
    correct_link = None
    for _, nearby in nearby_links.iterrows():
        if nearby['link_id'] != link_id and nearby.geometry.distance(poi_point) < 5 / 111000:
            correct_link = nearby
            break
    if correct_link is not None:
        return 2, f"Update LINK_ID to {correct_link['link_id']}"
    
    naming1 = streets_naming[streets_naming['link_id'] == link1['link_id']]
    naming2 = streets_naming[streets_naming['link_id'] == link2['link_id']]
    if naming1.empty or naming2.empty:
        logging.warning(f"Missing naming data for link1 {link1['link_id']} or link2 {link2['link_id']}")
        return 3, f"Set MULTIDIGIT = 'N' for links {link1['link_id']}, {link2['link_id']}"
    same_name = naming1.iloc[0]['ST_NAME'] == naming2.iloc[0]['ST_NAME']
    no_vegetation = not check_vegetation(link_id, poi_point)
    distance_ok = link1.geometry.distance(link2.geometry) < 30 / 111000
    if not (same_name and no_vegetation and distance_ok):
        return 3, f"Set MULTIDIGIT = 'N' for links {link1['link_id']}, {link2['link_id']}"
    
    return 4, "Legitimate Exception"

def apply_corrections(poi_gdf, streets_nav, violations):
    """Apply corrections based on scenario classification."""
    logging.info("Starting corrections")
    validations = []
    poi_gdf['EXCEPTION'] = None
    poi_gdf['PROCESSED'] = False
    poi_gdf['scenario'] = 0
    poi_gdf['action'] = 'None'
    
    for _, violation in violations.iterrows():
        poi_id = str(violation['POI_ID'])
        scenario = violation['scenario']
        action = violation['action']
        try:
            poi_idx = poi_gdf[poi_gdf['POI_ID'] == poi_id].index[0]
        except IndexError:
            logging.warning(f"POI {poi_id} not found in poi_gdf")
            continue
        
        poi_gdf.loc[poi_idx, 'PROCESSED'] = True
        poi_gdf.loc[poi_idx, 'scenario'] = scenario
        poi_gdf.loc[poi_idx, 'action'] = action
        
        if scenario == 1:
            poi_gdf = poi_gdf.drop(poi_idx)
            validations.append({'POI_ID': poi_id, 'Action': 'Deleted', 'scenario': scenario})
        elif scenario == 2:
            match = re.search(r'Update LINK_ID to (\d+)', action)
            if match:
                new_link_id = match.group(1)
                poi_gdf.loc[poi_idx, 'LINK_ID'] = int(new_link_id)
                validations.append({'POI_ID': poi_id, 'Action': f'Updated LINK_ID to {new_link_id}', 'scenario': scenario})
            else:
                logging.warning(f"Could not parse LINK_ID from action: {action}")
                validations.append({'POI_ID': poi_id, 'Action': 'Skipped: Invalid LINK_ID', 'scenario': scenario})
                continue
        elif scenario == 3:
            match = re.search(r'Set MULTIDIGIT = \'N\' for links (\d+), (\d+)', action)
            if match:
                link1_id, link2_id = match.group(1), match.group(2)
                link1_idx = streets_nav[streets_nav['link_id'] == int(link1_id)].index
                link2_idx = streets_nav[streets_nav['link_id'] == int(link2_id)].index
                if not link1_idx.empty and not link2_idx.empty:
                    streets_nav.loc[link1_idx[0], 'MULTIDIGIT'] = 'N'
                    streets_nav.loc[link2_idx[0], 'MULTIDIGIT'] = 'N'
                    validations.append({'POI_ID': poi_id, 'Action': f'Set MULTIDIGIT = N for links {link1_id}, {link2_id}', 'scenario': scenario})
                else:
                    logging.warning(f"Links {link1_id} or {link2_id} not found")
                    validations.append({'POI_ID': poi_id, 'Action': 'Skipped: Invalid links', 'scenario': scenario})
            else:
                logging.warning(f"Could not parse links from action: {action}")
                validations.append({'POI_ID': poi_id, 'Action': 'Skipped: Invalid action', 'scenario': scenario})
                continue
        elif scenario == 4:
            poi_gdf.loc[poi_idx, 'EXCEPTION'] = 'Y'
            validations.append({'POI_ID': poi_id, 'Action': 'Marked as Legitimate Exception', 'scenario': scenario})
        
        logging.info(f"POI {poi_id}: Scenario {scenario} - {action}")
    
    validations_df = pd.DataFrame(validations)
    if not validations_df.empty:
        validations_df['POI_ID'] = validations_df['POI_ID'].astype(str)
    logging.info(f"Completed corrections: {len(validations)} actions applied")
    logging.info(f"Validations columns: {list(validations_df.columns)}")
    return poi_gdf, streets_nav, validations_df

def validate_outputs(poi_gdf, validations_df, streets_nav, original_poi_count):
    """Validate output integrity."""
    logging.info(f"Validating outputs: {len(poi_gdf)} POIs, {len(validations_df)} actions")
    deleted = len(validations_df[validations_df['Action'] == 'Deleted']) if not validations_df.empty else 0
    try:
        assert len(poi_gdf) == original_poi_count - deleted, f"POI count mismatch: {len(poi_gdf)} vs {original_poi_count - deleted}"
        assert all(col in poi_gdf.columns for col in ['POI_ID', 'LINK_ID', 'PERCFRREF', 'POI_ST_SD', 'POI_NAME', 'EXCEPTION', 'PROCESSED', 'scenario', 'action']), "Missing columns in poi_gdf"
        if not validations_df.empty:
            assert all(col in validations_df.columns for col in ['POI_ID', 'Action', 'scenario']), "Missing columns in validations_df"
        assert all(streets_nav['MULTIDIGIT'].isin(['Y', 'N'])), "Invalid MULTIDIGIT values"
        logging.info("Output validation passed")
    except AssertionError as e:
        logging.error(f"Output validation failed: {e}")
        raise

def visualize_pois(pois_csv, validations_csv, streets_geojson, output_file='map.html'):
    """Generate folium map visualizing POIs between MULTIDIGIT roads with scenario coloring."""
    logging.info(f"Generating visualization to {output_file}")
    
    try:
        # Load validations
        validations_df = pd.read_csv(validations_csv)
        logging.info(f"Loaded validations: {len(validations_df)} records")
        
        if validations_df.empty:
            logging.warning(f"No validations found in {validations_csv}")
            print(f"No MULTIDIGIT violations to visualize. Skipping {output_file}.")
            return

        required_validation_cols = ['POI_ID', 'scenario', 'Action']
        missing_cols = [col for col in required_validation_cols if col not in validations_df.columns]
        if missing_cols:
            logging.error(f"Missing columns in validations: {missing_cols}")
            validations_df['scenario'] = validations_df.get('scenario', 0)
            validations_df['Action'] = validations_df.get('Action', 'None')

        validations_df['POI_ID'] = validations_df['POI_ID'].astype(str)
        
        # Load POIs
        pois = pd.read_csv(pois_csv)
        logging.info(f"Loaded POIs: {len(pois)} records")
        
        if not all(col in pois.columns for col in ['POI_ID', 'x', 'y']):
            raise ValueError("POIs CSV missing required columns (POI_ID, x, y)")
            
        pois['POI_ID'] = pois['POI_ID'].astype(str)
        pois = pois[pois['POI_ID'].isin(validations_df['POI_ID'])]
        if pois.empty:
            logging.warning(f"No matching POIs found between POIs and validations in {pois_csv}")
            print(f"No matching POIs to visualize. Skipping {output_file}.")
            return
            
        null_coords = pois['x'].isna() | pois['y'].isna()
        logging.info(f"Null coordinates in pois: {null_coords.sum()} ({null_coords.mean()*100:.2f}%)")
        pois = pois[~null_coords]
        if pois.empty:
            logging.warning(f"No POIs with valid coordinates to visualize in {pois_csv}")
            print(f"No POIs with valid coordinates to visualize. Skipping {output_file}.")
            return
        
        # Merge POIs with validations
        logging.info("Merging POIs with validations...")
        pois_gdf = pois.merge(
            validations_df[['POI_ID', 'scenario', 'Action']],
            on='POI_ID',
            how='left'
        )
        logging.info(f"Merged data: {len(pois_gdf)} records")
        
        # Ensure scenario and Action columns
        logging.info(f"pois_gdf dtypes before processing: {pois_gdf.dtypes.to_dict()}")
        if 'scenario' not in pois_gdf.columns:
            pois_gdf['scenario'] = 0
        pois_gdf['scenario'] = pois_gdf['scenario'].astype(int)
        pois_gdf['Action'] = pois_gdf['Action'].fillna('None')
        logging.info(f"pois_gdf scenario values: {pois_gdf['scenario'].unique()}")
        
        # Create GeoDataFrame
        logging.info("Creating GeoDataFrame...")
        pois_gdf = gpd.GeoDataFrame(
            pois_gdf,
            geometry=gpd.points_from_xy(pois_gdf['x'], pois_gdf['y']),
            crs="EPSG:4326"
        )
        
        null_geoms = pois_gdf.geometry.isna()
        logging.info(f"Null geometries in pois_gdf: {null_geoms.sum()} ({null_geoms.mean()*100:.2f}%)")
        pois_gdf = pois_gdf[~null_geoms]
        if pois_gdf.empty:
            logging.warning(f"No POIs with valid geometries to visualize in {pois_csv}")
            print(f"No POIs with valid coordinates to visualize. Skipping {output_file}.")
            return
        
        # Load streets
        streets = gpd.read_file(streets_geojson)
        logging.info(f"Loaded streets: {len(streets)} features")
        
        # Create visualization
        logging.info("Creating map visualization...")
        scenario_config = {
            1: {'color': 'black', 'icon': 'remove', 'name': 'Deleted'},
            2: {'color': 'blue', 'icon': 'random', 'name': 'Wrong LINK_ID'},
            3: {'color': 'orange', 'icon': 'road', 'name': 'Invalid MULTIDIGIT'},
            4: {'color': 'green', 'icon': 'ok-sign', 'name': 'Legitimate'},
            0: {'color': 'gray', 'icon': 'question-sign', 'name': 'Unknown'}
        }

        centroid = pois_gdf.geometry.union_all().centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)

        folium.GeoJson(
            streets,
            style_function=lambda x: {
                'color': 'blue' if x['properties'].get('MULTIDIGIT') == 'Y' else 'green',
                'weight': 4 if x['properties'].get('MULTIDIGIT') == 'Y' else 2,
                'opacity': 0.7
            },
            name='Road Network'
        ).add_to(m)

        for _, poi in pois_gdf.iterrows():
            scenario = int(poi['scenario'])
            config = scenario_config.get(scenario, scenario_config[0])
            
            popup_html = f"""
            <div style="width: 250px">
                <h4>{poi.get('POI_NAME', 'Unknown')}</h4>
                <hr>
                <p><b>POI_ID:</b> {poi['POI_ID']}</p>
                <p><b>Scenario:</b> {config['name']}</p>
                <p><b>Action:</b> {poi['Action']}</p>
                <p><b>Coordinates:</b> {poi.geometry.y:.6f}, {poi.geometry.x:.6f}</p>
            </div>
            """
            
            folium.Marker(
                location=[poi.geometry.y, poi.geometry.x],
                icon=folium.Icon(
                    color=config['color'],
                    icon=config['icon'],
                    prefix='glyphicon'
                ),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        HeatMap(
            [[poi.geometry.y, poi.geometry.x] for _, poi in pois_gdf.iterrows()],
            name='POI Density',
            radius=15,
            blur=10
        ).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_file)
        logging.info(f"Successfully saved visualization to {output_file}")
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Visualization failed for {output_file}: {e}")
        print(f"Failed to create visualization for {output_file}: {str(e)}")
        raise

def main():
    """Main function to process POI295 validations for test or full mode."""
    logging.info("GeoFix295 started")
    parser = argparse.ArgumentParser(description='Process POI295 validations')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='Mode: test (1 file per dataset) or full (all files)')
    args = parser.parse_args()
    
    poi_folder = 'POIs'
    streets_nav_folder = 'STREETS_NAV'
    streets_naming_folder = 'STREETS_NAMING_ADDRESSING'
    
    if args.mode == 'test':
        # Test mode: Process one tile
        print("Loading datasets...")
        poi_df, streets_nav, streets_naming = load_datasets(
            poi_folder, streets_nav_folder, streets_naming_folder, mode='test'
        )
        original_poi_count = len(poi_df)
        
        print(f"Computing POI geometries for {len(poi_df)} POIs...")
        poi_gdf = compute_poi_geometry(poi_df, streets_nav)
        
        print("Finding MULTIDIGIT pairs...")
        multidigit_pairs = find_multidigit_pairs(streets_nav, streets_naming)
        
        print(f"Detecting violations among {len(multidigit_pairs)} MULTIDIGIT pairs...")
        violations = []
        for i, (link_id1, link_id2, geom1, geom2) in enumerate(multidigit_pairs):
            pois = poi_gdf[poi_gdf['LINK_ID'].isin([link_id1, link_id2])]
            for _, poi in pois.iterrows():
                poi_point = poi.geometry
                if poi_point and check_inside_multidigit(poi_point, geom1, geom2, link_id1, link_id2):
                    scenario, action = classify_scenario(
                        poi,
                        poi_point,
                        streets_nav[streets_nav['link_id'] == link_id1].iloc[0],
                        streets_nav[streets_nav['link_id'] == link_id2].iloc[0],
                        streets_nav,
                        streets_naming
                    )
                    violations.append({
                        'POI_ID': str(poi['POI_ID']),
                        'link_id1': link_id1,
                        'link_id2': link_id2,
                        'scenario': scenario,
                        'action': action
                    })
            if i % 10 == 0 and i > 0:
                print(f"Processed {i}/{len(multidigit_pairs)} MULTIDIGIT pairs")
        
        print(f"Found {len(violations)} violations")
        logging.info(f"Detected {len(violations)} POI295 violations")
        
        print("Applying corrections...")
        updated_poi_gdf, updated_streets_nav, validations_df = apply_corrections(
            poi_gdf, streets_nav, pd.DataFrame(violations)
        )
        
        print("Validating outputs...")
        validate_outputs(updated_poi_gdf, validations_df, updated_streets_nav, original_poi_count)
        
        print("Saving outputs...")
        updated_poi_gdf['x'] = updated_poi_gdf['geometry'].x
        updated_poi_gdf['y'] = updated_poi_gdf['geometry'].y
        updated_poi_gdf.drop(columns=['geometry']).to_csv('updated_pois.csv', index=False)
        updated_streets_nav.to_file('updated_streets_nav.geojson', driver='GeoJSON')
        validations_df.to_csv('validations.csv', index=False)
        logging.info("Saved outputs: updated_pois.csv, updated_streets_nav.geojson, validations.csv")
        
        print("Generating visualization...")
        visualize_pois('updated_pois.csv', 'validations.csv', 'updated_streets_nav.geojson', output_file='map.html')
        
        print(f"Processed {len(violations)} POI295 violations. Outputs saved. Visualization in map.html")
    else:
        # Full mode: Process all tiles independently and combine results
        # Get tile IDs from POI files
        poi_files = [f for f in os.listdir(poi_folder) if f.endswith('.csv')]
        tile_ids = sorted({f.split('_')[-1].split('.')[0] for f in poi_files})
        print(f"Processing {len(tile_ids)} tiles in full mode")
        
        updated_pois = []
        updated_streets = []
        validations = []
        original_poi_count = 0
        
        for tile_id in tqdm(tile_ids, desc="Processing tiles"):
            print(f"\nProcessing tile {tile_id}")
            try:
                # Load data for this tile only
                poi_df, streets_nav, streets_naming = load_datasets(
                    poi_folder, streets_nav_folder, streets_naming_folder, 
                    mode='test', tile_id=tile_id
                )
                if poi_df.empty or streets_nav.empty or streets_naming.empty:
                    logging.warning(f"Empty dataset for tile {tile_id}, skipping")
                    continue
                
                tile_poi_count = len(poi_df)
                original_poi_count += tile_poi_count
                
                # Compute geometries
                print(f"Computing POI geometries for {tile_poi_count} POIs...")
                poi_gdf = compute_poi_geometry(poi_df, streets_nav)
                
                # Find MULTIDIGIT pairs
                print("Finding MULTIDIGIT pairs...")
                multidigit_pairs = find_multidigit_pairs(streets_nav, streets_naming)
                
                # Detect violations
                print(f"Detecting violations among {len(multidigit_pairs)} MULTIDIGIT pairs...")
                tile_violations = []
                for i, (link_id1, link_id2, geom1, geom2) in enumerate(multidigit_pairs):
                    pois = poi_gdf[poi_gdf['LINK_ID'].isin([link_id1, link_id2])]
                    for _, poi in pois.iterrows():
                        poi_point = poi.geometry
                        if poi_point and check_inside_multidigit(poi_point, geom1, geom2, link_id1, link_id2):
                            scenario, action = classify_scenario(
                                poi,
                                poi_point,
                                streets_nav[streets_nav['link_id'] == link_id1].iloc[0],
                                streets_nav[streets_nav['link_id'] == link_id2].iloc[0],
                                streets_nav,
                                streets_naming
                            )
                            tile_violations.append({
                                'POI_ID': str(poi['POI_ID']),
                                'link_id1': link_id1,
                                'link_id2': link_id2,
                                'scenario': scenario,
                                'action': action
                            })
                    if i % 10 == 0 and i > 0:
                        print(f"Processed {i}/{len(multidigit_pairs)} MULTIDIGIT pairs")
                
                print(f"Tile {tile_id}: Found {len(tile_violations)} violations")
                logging.info(f"Tile {tile_id}: Detected {len(tile_violations)} POI295 violations")
                
                # Apply corrections
                print("Applying corrections...")
                updated_poi_gdf, updated_streets_nav, validations_df = apply_corrections(
                    poi_gdf, streets_nav, pd.DataFrame(tile_violations)
                )
                
                # Validate tile outputs
                print("Validating tile outputs...")
                validate_outputs(updated_poi_gdf, validations_df, updated_streets_nav, tile_poi_count)
                
                # Save tile-specific outputs
                tile_poi_csv = f'updated_pois_tile_{tile_id}.csv'
                tile_validations_csv = f'validations_tile_{tile_id}.csv'
                tile_streets_geojson = f'updated_streets_nav_tile_{tile_id}.geojson'
                
                updated_poi_gdf['x'] = updated_poi_gdf['geometry'].x
                updated_poi_gdf['y'] = updated_poi_gdf['geometry'].y
                updated_poi_gdf.drop(columns=['geometry']).to_csv(tile_poi_csv, index=False)
                updated_streets_nav.to_file(tile_streets_geojson, driver='GeoJSON')
                validations_df.to_csv(tile_validations_csv, index=False)
                logging.info(f"Saved tile outputs: {tile_poi_csv}, {tile_validations_csv}, {tile_streets_geojson}")
                
                # Generate tile-specific map
                print("Generating tile visualization...")
                visualize_pois(
                    tile_poi_csv, 
                    tile_validations_csv, 
                    tile_streets_geojson, 
                    output_file=f'map_tile_{tile_id}.html'
                )
                
                # Collect results
                updated_pois.append(updated_poi_gdf)
                updated_streets.append(updated_streets_nav)
                validations.append(validations_df)
                
            except Exception as e:
                print(f"Error processing tile {tile_id}: {str(e)}")
                logging.error(f"Error processing tile {tile_id}: {e}")
                continue
        
        # Combine results
        print("\nCombining results from all tiles...")
        try:
            combined_poi_gdf = gpd.GeoDataFrame(
                pd.concat([df for df in updated_pois], ignore_index=True),
                crs="EPSG:4326"
            ).drop_duplicates(subset='POI_ID')
        except ValueError as e:
            logging.warning(f"No POIs to combine: {e}")
            combined_poi_gdf = gpd.GeoDataFrame(columns=updated_pois[0].columns if updated_pois else [], crs="EPSG:4326")
        
        try:
            combined_streets_nav = gpd.GeoDataFrame(
                pd.concat([df for df in updated_streets], ignore_index=True),
                crs="EPSG:4326"
            ).drop_duplicates(subset='link_id')
        except ValueError as e:
            logging.warning(f"No streets to combine: {e}")
            combined_streets_nav = gpd.GeoDataFrame(columns=updated_streets[0].columns if updated_streets else [], crs="EPSG:4326")
        
        combined_validations_df = pd.concat([df for df in validations], ignore_index=True).drop_duplicates(subset='POI_ID')
        if not combined_validations_df.empty:
            combined_validations_df['POI_ID'] = combined_validations_df['POI_ID'].astype(str)
        
        # Validate combined outputs
        print("Validating combined outputs...")
        validate_outputs(combined_poi_gdf, combined_validations_df, combined_streets_nav, original_poi_count)
        
        # Save combined outputs
        print("Saving combined outputs...")
        if not combined_poi_gdf.empty:
            combined_poi_gdf['x'] = combined_poi_gdf['geometry'].x
            combined_poi_gdf['y'] = combined_poi_gdf['geometry'].y
            combined_poi_gdf.drop(columns=['geometry']).to_csv('updated_pois.csv', index=False)
        else:
            logging.warning("No POIs to save to updated_pois.csv")
            pd.DataFrame().to_csv('updated_pois.csv', index=False)
        
        if not combined_streets_nav.empty:
            combined_streets_nav.to_file('updated_streets_nav.geojson', driver='GeoJSON')
        else:
            logging.warning("No streets to save to updated_streets_nav.geojson")
            gpd.GeoDataFrame().to_file('updated_streets_nav.geojson', driver='GeoJSON')
        
        combined_validations_df.to_csv('validations.csv', index=False)
        logging.info("Saved combined outputs: updated_pois.csv, validations.csv, updated_streets_nav.geojson")
        
        print(f"Processed {len(combined_validations_df)} POI295 violations. Combined outputs saved.")
        print(f"Per-tile maps saved as map_tile_<tile_id>.html")
        logging.info(f"GeoFix295 completed: {len(combined_validations_df)} violations processed")
        if not combined_validations_df.empty:
            logging.info(f"Stats - Deleted: {len(combined_validations_df[combined_validations_df['Action'] == 'Deleted'])}, "
                         f"Relocated: {len(combined_validations_df[combined_validations_df['Action'].str.contains('Updated LINK_ID')])}, "
                         f"Roads Corrected: {len(combined_validations_df[combined_validations_df['Action'].str.contains('Set MULTIDIGIT')])}, "
                         f"Legitimate: {len(combined_validations_df[combined_validations_df['Action'] == 'Marked as Legitimate Exception'])}")

if __name__ == '__main__':
    main()