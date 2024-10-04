
# Data paths and attributes
BUILDING_DATA_PATH = 'data/raw/OSM_buildings_in_singapore_epsg4326.geojson'
LAND_PLOT_DATA_PATH = 'data/raw/MasterPlan2019LandUselayer.geojson'
LAND_PLOT_ATTRIBUTES = ['LU_DESC']
BUILDING_PLOT_DATA_PATH = 'data/cleaned/buildings_with_plot.geojson'
DISTANCE_MATRIX_PATH = 'data/cleaned/distance_matrix.csv'
CLUSTERED_BUILDING_DATA_PATH = 'data/clustered/clustered_buildings.geojson'
REPRESENTATIVE_DISTRICT_DIRECTORY = 'data/clustered/representative_districts'
CEA_FILES_DIRECTORY = 'data/cea'
NBR_BUILDINGS = 'all'

# Footprint clustering parameters
CLUSTERING_APPROACH = 'hdbscan'
POLYGON_DISTANCE_METRIC = 'hausdorff'
DBSCAN_EPS = 180    # meters
DBSCAN_MIN_SAMPLES = 3
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 3

# Land use clustering parameters
KMEDOIDS_K = 5
LAND_USE_DISTANCE_METRIC = 'manhattan'

### Hyperparameter tuning ###
HYPERPARAMETER_TUNING = True

# Footprint clustering hyperparameter grid search:
#   DBSCAN
MIN_DBSCAN_EPS = 40
MAX_DBSCAN_EPS = 200
DELTA_DBSCAN_EPS = 20

MIN_DBSCAN_MIN_SAMPLES = 10
MAX_DBSCAN_MIN_SAMPLES = 30
DELTA_DBSCAN_MIN_SAMPLES = 2

#   HDBSCAN
MIN_HDBSCAN_MIN_CLUSTER_SIZE = 10
MAX_HDBSCAN_MIN_CLUSTER_SIZE = 50
DELTA_HDBSCAN_MIN_CLUSTER_SIZE = 4

MIN_HDBSCAN_MIN_SAMPLES = 3
MAX_HDBSCAN_MIN_SAMPLES = 10
DELTA_HDBSCAN_MIN_SAMPLES = 1

# Land use clustering hyperparameters
MIN_LAND_USE_CLUSTERS = 3
MAX_LAND_USE_CLUSTERS = 20

# CEA data preparation
OVERLAP_THRESHOLD = 0.95
PLOT_MATCHED_BUILDINGS = True
