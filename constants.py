
# Data paths and attributes
BUILDING_DATA_PATH = 'data/raw/OSM_buildings_in_singapore_epsg4326.geojson'
LAND_PLOT_DATA_PATH = 'data/raw/MasterPlan2019LandUselayer.geojson'
LAND_PLOT_ATTRIBUTES = ['LU_DESC']
BUILDING_PLOT_DATA_PATH = 'data/cleaned/buildings_with_plot.geojson'
DISTANCE_MATRIX_PATH = 'data/cleaned/distance_matrix.csv'
CLUSTERED_BUILDING_DATA_PATH = 'data/clustered/clustered_buildings.geojson'
REPRESENTATIVE_DISTRICT_PATH = 'data/clustered/representative_buildings.shp'
NBR_BUILDINGS = 'all'

# Clustering parameters
CLUSTERING_APPROACH = 'hdbscan'
POLYGON_DISTANCE_METRIC = 'hausdorff'
N_LAND_USE_CLUSTERS = 5
LAND_USE_DISTANCE_METRIC = 'euclidean'
DBSCAN_EPS = 100    # meters
DBSCAN_MIN_SAMPLES = 5

