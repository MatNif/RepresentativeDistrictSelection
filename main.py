import geopandas as gpd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

from constants import BUILDING_PLOT_DATA_PATH, POLYGON_DISTANCE_METRIC, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, \
                      CLUSTERED_BUILDING_DATA_PATH, DISTANCE_MATRIX_PATH, NBR_BUILDINGS


def create_geographical_clusters():
    """
    Create geographical clusters of buildings based on the land plot they belong to:

    1. Group the buildings by land plot and compute a proxy for each land plot (e.g., convex hull).
    2. Calculate the distance matrix between the proxies.
    3. Apply DBSCAN to the distance matrix to cluster the land plots.
    4. Map the clusters back to the original buildings.

    :return:
    """
    # Load GeoJSON files
    if isinstance(NBR_BUILDINGS, int):
        buildings_with_plot = gpd.read_file(BUILDING_PLOT_DATA_PATH, rows=NBR_BUILDINGS)
    elif NBR_BUILDINGS == 'all':
        buildings_with_plot = gpd.read_file(BUILDING_PLOT_DATA_PATH)
    else:
        raise ValueError("NBR_BUILDINGS must be an integer or 'all'.")

    # Step 1: Compute a proxy geometry for each land plot
    plot_proxies = buildings_with_plot.groupby('Name', group_keys=False).apply(compute_plot_proxy, include_groups=False)
    plot_proxies_gdf = gpd.GeoDataFrame(plot_proxies, columns=['geometry'], crs=buildings_with_plot.crs)

    # Step 2: Calculate the distance matrix between the multipolygons
    distance_matrix = generate_distance_matrix(plot_proxies_gdf, metric=POLYGON_DISTANCE_METRIC)

    # Step 3: Apply DBSCAN using the distance matrix
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
    plot_proxies_gdf['cluster'] = dbscan.fit_predict(distance_matrix)

    # Step 4: Map the DBSCAN clusters back to the original buildings
    buildings_with_clusters = buildings_with_plot.merge(plot_proxies_gdf[['cluster']], left_on='Name', right_index=True)

    # Save the clusters to a GeoJSON file
    buildings_with_clusters.to_file(CLUSTERED_BUILDING_DATA_PATH, driver='GeoJSON')

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    buildings_with_clusters.plot(column='cluster', cmap='Set3', legend=True, ax=ax)
    ax.set_title('DBSCAN Clustering of Multipolygon Proxies')
    plt.show()
    input("Press Enter to continue...")

def compute_plot_proxy(buildings_in_plot):
    """
    Compute a proxy geometry for each plot that takes into account building numbers and locations within the plot.
    In this case the proxy is the convex hull of the buildings in the plot.

    :param buildings_in_plot: GeoDataFrame of buildings in the plot
    :return:
    """
    convex_hull = buildings_in_plot.union_all().convex_hull

    return convex_hull


def generate_distance_matrix(geometries_gdf, metric='euclidean'):
    """
    Generate a distance matrix between geometries using the specified metric.

    :param geometries_gdf: List of geometries
    :param metric: Distance metric to use
    :return: Distance matrix
    """
    # Check if the distance matrix already exists
    full_distance_matrix_path = Path(DISTANCE_MATRIX_PATH.split('.')[0] + f"_{metric}_{NBR_BUILDINGS}.h5")
    if full_distance_matrix_path.exists():
        # Load from HDF5
        with h5py.File(full_distance_matrix_path, 'r') as f:
            loaded_array = f['distance_matrix'][:]

        return loaded_array

    # Option 1: Use centroid distance
    if metric == 'centroid_euclidean':
        centroids = np.array([(geom.centroid.x, geom.centroid.y) for geom in geometries_gdf.geometry])
        distance_matrix = cdist(centroids, centroids, metric='euclidean')
    # Option 2: Use Hausdorff distance (requires a custom function)
    elif metric == 'hausdorff':
        distance_matrix = np.array([[geom1.hausdorff_distance(geom2) for geom2 in geometries_gdf.geometry]
                                    for geom1 in geometries_gdf.geometry])
    else:
        raise ValueError(f"Invalid distance metric: {metric}")

    # Save the distance matrix to a h5 file if it doesn't exist
    with h5py.File(full_distance_matrix_path, 'w') as f:
        f.create_dataset('distance_matrix', data=distance_matrix)

    return distance_matrix


# Entry point of the script
if __name__ == '__main__':
    create_geographical_clusters()
