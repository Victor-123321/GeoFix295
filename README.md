# GeoFix295: POI295 Validation and Correction Tool

## Overview
**GeoFix295** is a Python-based solution developed for the HERE Technologies GuadalaHacks Student Hackathon 2025. It addresses the POI295 validation challenge by detecting and correcting violations in Point of Interest (POI) data linked to multi-digitized road segments in Mexico City. The tool processes ~190,000 POIs across 20 tiles, identifying ~1000 violations (~50 per tile) where POIs are incorrectly placed between `MULTIDIGIT` road segments with the same `ST_NAME`. It classifies violations into four scenarios, applies corrections, and generates per-tile visualizations for efficient analysis, ensuring accurate POI placement in HERE’s map database.

## Features
- **Data Loading**: Loads POI CSVs, Streets Navigation GeoJSONs, and Streets Naming GeoJSONs for 20 tiles (#1).
- **Geometry Computation**: Calculates POI coordinates using `LINK_ID`, `PERCFRREF`, and `POI_ST_SD` (#2).
- **Violation Detection**: Identifies POIs between `MULTIDIGIT` road pairs with matching `ST_NAME` using R-tree spatial indexing (#3).
- **Scenario Classification**: Categorizes violations into four scenarios: non-existent, incorrect location, incorrect `MULTIDIGIT`, or legitimate exception (#4).
- **Corrections**: Applies fixes (delete POIs, update `LINK_ID`, set `MULTIDIGIT = 'N'`, mark exceptions) and saves outputs (`updated_pois.csv`, `validations.csv`, `updated_streets_nav.geojson`) (#4).
- **Visualization**: Generates per-tile Folium maps (`map_tile_<tile_id>.html`) showing roads, POI markers, and heatmaps (#25).
- **Full Mode**: Processes all tiles independently, combining results while avoiding performance issues with large datasets (#25).
- **Robustness**: Handles missing files, null geometries, and errors, with detailed logging (`poi295_corrections.log`) (#25).

## Algorithms and Implementation

### 1. `load_datasets`
- **Purpose**: Loads POI CSVs (`POIs/*.csv`), Streets Navigation GeoJSONs (`STREETS_NAV/*.geojson`), and Streets Naming GeoJSONs (`STREETS_NAMING_ADDRESSING/*.geojson`).
- **Logic**:
  - In `test` mode, loads one file per dataset.
  - In `full` mode, processes one tile at a time (filtered by `tile_id`), loading matching files.
  - Uses `pandas` for CSVs and `geopandas` for GeoJSONs, standardizing `POI_ID` and `link_id` as strings.
  - Handles empty datasets with empty DataFrames/GeoDataFrames.
- **Reason**: Modular loading ensures scalability for n tiles, with tile-specific filtering to replicate test mode in full mode, minimizing memory usage.

### 2. `compute_poi_geometry`
- **Purpose**: Computes POI coordinates from `LINK_ID`, `PERCFRREF`, and `POI_ST_SD`.
- **Logic**:
  - Matches each POI’s `LINK_ID` to a road segment in `streets_nav`.
  - Interpolates along the road geometry using `PERCFRREF` (normalized 0–1).
  - Applies a 5m offset left (`POI_ST_SD = 'L'`) or right (`'R'`) using a 90° angle.
  - Creates a `geopandas.GeoDataFrame` with `EPSG:4326` CRS.
  - Logs null geometries (e.g., missing `LINK_ID` or invalid road geometry).
- **Reason**: Accurate POI placement is critical for violation detection. Offsets simulate real-world positioning, and null geometry logging aids debugging.

### 3. `find_multidigit_pairs`
- **Purpose**: Identifies pairs of `MULTIDIGIT = 'Y'` road segments with the same `ST_NAME` within 50m.
- **Logic**:
  - Filters `streets_nav` for `MULTIDIGIT = 'Y'`.
  - Uses `rtree` spatial index to find nearby road pairs based on geometry bounds.
  - Matches `ST_NAME` from `streets_naming` and checks distance (<50m).
  - Returns pairs as `(link_id1, link_id2, geom1, geom2)`.
- **Reason**: R-tree optimizes spatial queries for large datasets. The 50m threshold captures relevant road pairs (e.g., divided highways), aligning with POI295 rules.

### 4. `check_inside_multidigit`
- **Purpose**: Checks if a POI lies between two `MULTIDIGIT` road segments.
- **Logic**:
  - Buffers each road by 10m and computes their union (or convex hull if not a polygon).
  - Tests if the POI’s point geometry is contained within the buffered polygon.
  - Validates inputs (non-null, valid geometries) and checks road separation (>1m).
- **Reason**: The buffered polygon approximates the area between roads (e.g., median strips), ensuring accurate violation detection. Robust checks prevent false positives.

### 5. `classify_scenario`
- **Purpose**: Classifies POI295 violations into four scenarios.
- **Logic**:
  - **Scenario 1 (Deleted)**: POI doesn’t exist (mock `check_poi_existence` returns False).
  - **Scenario 2 (Wrong LINK_ID)**: POI is <5m from a different road’s geometry.
  - **Scenario 3 (Invalid MULTIDIGIT)**: Roads lack same `ST_NAME`, have vegetation (mock `check_vegetation`), or are >30m apart; sets `MULTIDIGIT = 'N'`.
  - **Scenario 4 (Legitimate)**: Valid exception if all conditions (same `ST_NAME`, no vegetation, <30m) are met.
- **Reason**: Scenario-based classification ensures targeted corrections, balancing automated fixes with manual review for legitimate cases.

### 6. `apply_corrections`
- **Purpose**: Applies corrections based on scenario classification.
- **Logic**:
  - Deletes POIs (Scenario 1).
  - Updates `LINK_ID` (Scenario 2).
  - Sets `MULTIDIGIT = 'N'` for road pairs (Scenario 3).
  - Marks POIs as exceptions (`EXCEPTION = 'Y'`) (Scenario 4).
  - Tracks actions in `validations_df` and updates `poi_gdf`, `streets_nav`.
- **Reason**: Automated corrections streamline data cleaning, while `validations_df` provides an audit trail for transparency.

### 7. `visualize_pois`
- **Purpose**: Generates Folium maps (`map_tile_<tile_id>.html`) for violations.
- **Logic**:
  - Merges `updated_pois.csv` with `validations.csv` on `POI_ID`.
  - Creates a `geopandas.GeoDataFrame` from `x`, `y` coordinates.
  - Plots roads (`streets_nav`) with `MULTIDIGIT = 'Y'` in blue, others in green.
  - Adds POI markers colored by scenario (black: Deleted, blue: Wrong LINK_ID, orange: Invalid MULTIDIGIT, green: Legitimate).
  - Includes a heatmap for POI density and popups with `POI_ID`, `POI_NAME`, `Action`, and coordinates.
  - Saves maps per tile to avoid slowdowns.
- **Reason**: Folium provides interactive visualizations, critical for hackathon presentation (#22). Per-tile maps ensure performance even with a big number of violations.

### 8. `main`
- **Purpose**: Orchestrates the pipeline in `test` or `full` mode.
- **Logic**:
  - **Test Mode**: Processes one tile, producing `map.html` and combined outputs.
  - **Full Mode**: Iterates over the tiles, processing each one, saving per-tile outputs (`updated_pois_tile_<tile_id>.csv`, etc.) and maps (`map_tile_<tile_id>.html`), then combines results (`updated_pois.csv`, etc.) with duplicate removal (`POI_ID`, `link_id`).
  - Uses `tqdm` for progress tracking.
  - Validates outputs (`validate_outputs`) to ensure POI counts and column integrity.
- **Reason**: Full mode replicates test mode’s reliability  while scaling to n tiles. Duplicate removal ensures data consistency.

## Design Choices
- **Per-Tile Processing**: Full mode processes tiles independently to replicate test mode’s accuracy and manage memory for ~190,000 POIs (#25).
- **Folium Visualizations**: Interactive maps with scenario-colored markers and heatmaps enhance presentation impact and debugging (#22, #25).
- **R-tree Indexing**: Optimizes spatial queries in `find_multidigit_pairs`, crucial for large road networks (#3).
- **Robust Error Handling**: Skips failed tiles, logs null geometries, and handles empty datasets to ensure partial failures don’t halt execution (#25).
- **Mock Functions**: `check_poi_existence` and `check_vegetation` simulate external checks, allowing focus on geospatial logic within hackathon constraints.
- **Logging**: Detailed `poi295_corrections.log` tracks violations, null geometries, and errors, aiding debugging and transparency.

## Prerequisites
- **Python**: 3.9+
- **Dependencies**:
  ```bash
  pip install pandas geopandas shapely rtree folium numpy tqdm
  ```
- **Data** (n tiles, matching by `tile_id`, e.g., `tile_001.csv`, `tile_001.geojson`):
  - POI CSVs in `POIs/`
  - Streets Navigation GeoJSONs in `STREETS_NAV/`
  - Streets Naming GeoJSONs in `STREETS_NAMING_ADDRESSING/`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Victor-123321/GeoFix295.git
   cd GeoFix295
   ```
2. Organize data in the folder structure above.

## Usage
- **Test Mode** (1 tile):
  ```bash
  python poi295_correction.py --mode test
  ```
  - **Outputs**:
    - `updated_pois.csv`: Updated POIs with `x`, `y` coordinates.
    - `validations.csv`: Violation details (`POI_ID`, `scenario`, `Action`).
    - `updated_streets_nav.geojson`: Updated road segments.
    - `map.html`: Folium map with ~50 violations.
    - `poi295_corrections.log`: Detailed logs.

- **Full Mode** (n tiles):
  ```bash
  python poi295_correction.py --mode full
  ```
  - **Outputs**:
    - **Per-tile**:
      - `updated_pois_tile_<tile_id>.csv`
      - `validations_tile_<tile_id>.csv`
      - `updated_streets_nav_tile_<tile_id>.geojson`
      - `map_tile_<tile_id>.html` (~50 violations each)
    - **Combined**:
      - `updated_pois.csv`
      - `validations.csv`
      - `updated_streets_nav.geojson`
    - `poi295_corrections.log`

## Project Structure
```
GeoFix295/
├── POIs/                           # POI CSV files
├── STREETS_NAV/                    # Streets Navigation GeoJSONs
├── STREETS_NAMING_ADDRESSING/      # Streets Naming GeoJSONs
├── poi295_correction.py            # Main script
├── requirements.txt                # Dependencies
├── poi295_corrections.log          # Execution logs
├── updated_pois.csv                # Combined updated POIs
├── validations.csv                 # Combined violation details
├── updated_streets_nav.geojson     # Combined updated roads
├── updated_pois_tile_<tile_id>.csv # Per-tile updated POIs
├── validations_tile_<tile_id>.csv  # Per-tile violation details
├── updated_streets_nav_tile_<tile_id>.geojson # Per-tile updated roads
├── map_tile_<tile_id>.html         # Per-tile Folium maps
├── map.html                        # Test mode Folium map
└── README.md                       # This file
```

## Team Roles
- **Víctor Velázquez (Data Engineer)**: Data loading (`load_datasets`), geometry computation (`compute_poi_geometry`), full mode implementation.
- **Diego Lizárraga (GIS Specialist)**: Violation detection (`find_multidigit_pairs`, `check_inside_multidigit`), geospatial visualization (`visualize_pois`).
- **Sofía López (Project Lead)**: Scenario classification (`classify_scenario`), corrections (`apply_corrections`), documentation, bug fixes .

## Development Status
- **Completed Issues**:
  - #1: Data loading for all datasets.
  - #2: POI geometry computation.
  - #3: Violation detection with R-tree indexing.
  - #4: Scenario classification and corrections.
  - #25: Full mode for 20 tiles, per-tile maps, fixed `KeyError: 'Action'` and map generation error (`'int' object has no attribute 'fillna'`).
- **Fixed Bugs**:
  - `KeyError: 'Action'` in `visualize_pois` by ensuring column presence (#25).
  - Map generation error by handling integer `scenario` column (#25).

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add feature #issue-number"`.
4. Push: `git push origin feature/your-feature`.
5. Open a pull request to `develop`.

## License
MIT License. See `LICENSE`.

## Acknowledgments
- HERE Technologies for providing datasets and hosting the hackathon.
- GuadalaHacks 2025 organizers for the platform and support.