
# Data paths and attributes
BUILDING_DATA_PATH = 'data/raw/OSM_buildings_in_singapore_epsg4326.geojson'
LAND_PLOT_DATA_PATH = 'data/raw/MasterPlan2019LandUselayer.geojson'
LAND_PLOT_ATTRIBUTES = ['LU_DESC']
BUILDING_PLOT_DATA_PATH = 'data/cleaned/buildings_with_plot.geojson'
DISTANCE_MATRIX_PATH = 'data/cleaned/distance_matrix.csv'
CLUSTERED_BUILDING_DATA_PATH = 'data/clustered/clustered_buildings.geojson'
NBR_BUILDINGS = 'all'

# Clustering parameters
POLYGON_DISTANCE_METRIC = 'hausdorff'
DBSCAN_EPS = 100    # meters
DBSCAN_MIN_SAMPLES = 5

