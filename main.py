import geopandas as gpd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import hdbscan
from kmedoids import KMedoids

from constants import BUILDING_PLOT_DATA_PATH, POLYGON_DISTANCE_METRIC, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, \
                      CLUSTERING_APPROACH, CLUSTERED_BUILDING_DATA_PATH, DISTANCE_MATRIX_PATH, NBR_BUILDINGS, \
                      N_LAND_USE_CLUSTERS, LAND_USE_DISTANCE_METRIC, REPRESENTATIVE_DISTRICT_PATH


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

    # Step 3: Apply chosen clustering approach (and make use of the distance matrix)
    plot_proxies_gdf['cluster'] = apply_clustering_approach(CLUSTERING_APPROACH, distance_matrix=distance_matrix)

    # Step 4: Map the DBSCAN clusters back to the original buildings
    buildings_with_clusters = buildings_with_plot.merge(plot_proxies_gdf[['cluster']], left_on='Name', right_index=True)

    # Save the clusters to a GeoJSON file
    buildings_with_clusters.to_file(CLUSTERED_BUILDING_DATA_PATH, driver='GeoJSON')

    # Visualization

    fig, ax = plt.subplots(figsize=(10, 10))
    buildings_with_clusters.plot(column='cluster', cmap='Set3', legend=True, ax=ax)
    ax.set_title('DBSCAN Clustering of Multipolygon Proxies')
    plt.show()


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
        distance_matrix = cdist(XA=centroids, XB=centroids, metric='euclidean')
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


def apply_clustering_approach(approach, distance_matrix=None):
    """
    Apply the chosen clustering approach to the distance matrix.

    :param approach: Clustering approach to use
    :param distance_matrix: Distance matrix to use
    :return: Clusters
    """
    if approach == 'dbscan':
        clusterer = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
        return clusterer.fit_predict(distance_matrix)
    elif approach == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=4, metric='precomputed')
        return clusterer.fit_predict(distance_matrix)
    else:
        raise ValueError(f"Invalid clustering approach: {approach}")


def select_representative_districts():
    """
    Select a representative district from each cluster based on a chosen criterion (e.g., building count).

    :return:
    """
    # Load the clustered building data
    buildings_with_clusters = gpd.read_file(CLUSTERED_BUILDING_DATA_PATH)

    # Step 1: Group by 'cluster' and 'LU_DESC' to get counts of land-use types per cluster
    cluster_landuse_counts = buildings_with_clusters.groupby(['cluster', 'LU_DESC']).size().unstack(fill_value=0)[1:]

    # Step 2: Calculate the percentage of each land-use type within each cluster
    cluster_landuse_percentage = cluster_landuse_counts.div(cluster_landuse_counts.sum(axis=1), axis=0)
    cluster_landuse_array = cluster_landuse_percentage.fillna(0).to_numpy()  # Ensure no NaN values

    # Step 3: Perform k-medoid clustering based on the percentage distributions
    kmedoids = KMedoids(N_LAND_USE_CLUSTERS, method='fasterpam', metric=LAND_USE_DISTANCE_METRIC, random_state=42)
    district_fit = kmedoids.fit(cluster_landuse_array)

    # Step 4: Identify the representative districts
    representative_districts = buildings_with_clusters[buildings_with_clusters['cluster'].isin(district_fit.medoid_indices_)]

    # Save the representative districts to a Shape-file
    for i, medoid_idx in enumerate(district_fit.medoid_indices_):
        representative_district = representative_districts[representative_districts['cluster'] == medoid_idx]
        representative_district.to_file(REPRESENTATIVE_DISTRICT_PATH, driver='ESRI Shapefile')

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    buildings_with_clusters.plot(column='cluster', cmap='Set3', legend=True, ax=ax)
    representative_districts.plot(ax=ax, color='black', edgecolor='black', linewidth=2)
    ax.set_title('Representative Districts')
    plt.show()


# Entry point of the script
if __name__ == '__main__':
    create_geographical_clusters()
    select_representative_districts()
