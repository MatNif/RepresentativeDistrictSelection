
# Data paths and attributes
BUILDING_DATA_PATH = 'data/raw/OSM_buildings_in_singapore_epsg4326.geojson'
LAND_PLOT_DATA_PATH = 'data/raw/MasterPlan2019LandUselayer.geojson'
LAND_PLOT_ATTRIBUTES = ['LU_DESC']
BUILDING_PLOT_DATA_PATH = 'data/cleaned/buildings_with_plot.geojson'
DISTANCE_MATRIX_PATH = 'data/cleaned/distance_matrix.csv'
CLUSTERED_BUILDING_DATA_PATH = 'data/clustered/clustered_buildings.geojson'
REPRESENTATIVE_DISTRICT_DIRECTORY = 'data/clustered/representative_districts'
NBR_BUILDINGS = 'all'

# Clustering parameters
CLUSTERING_APPROACH = 'hdbscan'
POLYGON_DISTANCE_METRIC = 'hausdorff'
DBSCAN_EPS = 180    # meters
DBSCAN_MIN_SAMPLES = 3
HDBSCAN_MIN_CLUSTER_SIZE = 14
HDBSCAN_MIN_SAMPLES = 3

MIN_LAND_USE_CLUSTERS = 2
MAX_LAND_USE_CLUSTERS = 20
LAND_USE_DISTANCE_METRIC = 'manhattan'

HYPERPARAMETER_TUNING = True
