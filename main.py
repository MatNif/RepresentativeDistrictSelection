import geopandas as gpd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import numpy as np
from pathlib import Path
import pandas as pd
import math
import os
import shutil
import h5py
import matplotlib.pyplot as plt
import hdbscan
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score

from constants import BUILDING_PLOT_DATA_PATH, POLYGON_DISTANCE_METRIC, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, \
                      CLUSTERING_APPROACH, CLUSTERED_BUILDING_DATA_PATH, DISTANCE_MATRIX_PATH, NBR_BUILDINGS, \
                      LAND_USE_DISTANCE_METRIC, REPRESENTATIVE_DISTRICT_DIRECTORY, HDBSCAN_MIN_CLUSTER_SIZE, \
                      HDBSCAN_MIN_SAMPLES, MIN_LAND_USE_CLUSTERS, MAX_LAND_USE_CLUSTERS, HYPERPARAMETER_TUNING


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
    plot_proxies_gdf['cluster'] = apply_clustering_approach(CLUSTERING_APPROACH, distance_matrix=distance_matrix,
                                                            analyse_hyperparameters=HYPERPARAMETER_TUNING)

    # Step 4: Map the clusters back to the original buildings
    buildings_with_clusters = buildings_with_plot.merge(plot_proxies_gdf[['cluster']], left_on='Name', right_index=True)

    # Save the clusters to a GeoJSON file
    buildings_with_clusters.to_file(CLUSTERED_BUILDING_DATA_PATH, driver='GeoJSON')

    # Visualization
    visualize_clusters(buildings_with_clusters)


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


def apply_clustering_approach(approach, distance_matrix=None, analyse_hyperparameters=False):
    """
    Apply the chosen clustering approach to the distance matrix.

    :param approach: Clustering approach to use
    :param distance_matrix: Distance matrix to use
    :param analyse_hyperparameters: Whether to analyse hyperparameters
    :return: Clusters
    """
    if not analyse_hyperparameters:
        clusterer = define_clusterer(approach)
        clusters = clusterer.fit_predict(distance_matrix)
        return clusters
    else:
        best_clusters = analyse_hyperparameters_for_clustering(approach, distance_matrix)
        return best_clusters


def analyse_hyperparameters_for_clustering(approach, distance_matrix):
    """
    Analyse the hyperparameters for the clustering approach and return the best clusters.
    """
    if approach == 'dbscan':
        return analyse_hyperparameters_for_dbscan(distance_matrix)
    elif approach == 'hdbscan':
        return analyse_hyperparameters_for_hdbscan(distance_matrix)


def analyse_hyperparameters_for_dbscan(distance_matrix):
    """
    Analyse the hyperparameters for DBSCAN and return the best clusters.
    """

    # Define hyperparameter ranges
    min_dbscan_eps = 40
    max_dbscan_eps = 200
    delta_dbscan_eps = 20

    min_dbscan_min_samples = 10
    max_dbscan_min_samples = 30
    delta_dbscan_min_samples = 2

    # Initialize the best hyperparameters and best clusters
    best_share_outliers = 1
    lowest_share_outliers = 1
    best_cluster_size_cv = np.inf
    lowest_cluster_size_cv = np.inf
    best_eps = 0
    best_min_samples = 0
    best_clusters = None

    # Grid search over hyperparameters
    for dbscan_eps in range(min_dbscan_eps, max_dbscan_eps + 1, delta_dbscan_eps):
        for dbscan_min_samples in range(min_dbscan_min_samples, max_dbscan_min_samples + 1, delta_dbscan_min_samples):

            # Perform clustering
            clusterer = define_clusterer("dbscan", dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples)
            clusters = clusterer.fit_predict(distance_matrix)

            # Calculate the clustering metrics
            share_outliers = np.sum(clusters == -1) / len(clusters)
            buildings_per_cluster = pd.Series(clusters[clusters != -1]).value_counts()
            cluster_size_cv = float(buildings_per_cluster.std() / buildings_per_cluster.mean())

            # Print cluster metrics evaluation
            print(f"Coefficient of variation of cluster size: {cluster_size_cv:.2f}, "
                  f"Share of outliers: {share_outliers:.2f}, "
                  f"eps: {dbscan_eps}, min_samples: {dbscan_min_samples}")

            if share_outliers < lowest_share_outliers:
                print(f"New lowest share of outliers: {share_outliers:.2f}")
                lowest_share_outliers = share_outliers

            if cluster_size_cv < lowest_cluster_size_cv:
                print(f"New lowest coefficient of variation of cluster size: {cluster_size_cv:.2f}")
                lowest_cluster_size_cv = cluster_size_cv

            # Select the best cluster hyperparameters based on the euclidean distance to the dream point in the
            # hyperparameter space drawn by the lowest share of outliers and the lowest coefficient of variation of
            # cluster size
            this_dream_point_distance = math.sqrt((share_outliers - lowest_share_outliers) ** 2 +
                                                  (cluster_size_cv - lowest_cluster_size_cv) ** 2)
            best_dream_point_distance = math.sqrt((best_share_outliers - lowest_share_outliers) ** 2 +
                                                  (best_cluster_size_cv - lowest_cluster_size_cv) ** 2)
            if this_dream_point_distance < best_dream_point_distance:
                best_share_outliers = share_outliers
                best_cluster_size_cv = cluster_size_cv
                best_eps = dbscan_eps
                best_min_samples = dbscan_min_samples
                best_clusters = clusters

    print(f"Best clusters -- coefficient of variation of cluster size: {best_cluster_size_cv:.2f}, "
          f"share of outliers: {best_share_outliers:.2f}, "
          f"eps: {best_eps}, min_samples: {best_min_samples}")

    return best_clusters


def analyse_hyperparameters_for_hdbscan(distance_matrix):
    # Define hyperparameter ranges
    min_hdbscan_min_cluster_size = 10
    max_hdbscan_min_cluster_size = 50
    delta_hdbscan_min_cluster_size = 4

    min_hdbscan_min_samples = 3
    max_hdbscan_min_samples = 10
    delta_hdbscan_min_samples = 1

    # Initialize the best hyperparameters and best clusters
    best_share_outliers = 1
    lowest_share_outliers = 1
    best_cluster_size_cv = np.inf
    lowest_cluster_size_cv = np.inf
    best_min_samples = 0
    best_min_cluster_size = 0
    best_clusters = None

    # Grid search over hyperparameters
    for hdbscan_min_cluster_size in range(min_hdbscan_min_cluster_size, max_hdbscan_min_cluster_size + 1,
                                          delta_hdbscan_min_cluster_size):
        for hdbscan_min_samples in range(min_hdbscan_min_samples, max_hdbscan_min_samples + 1,
                                         delta_hdbscan_min_samples):
            # Perform clustering
            clusterer = define_clusterer("hdbscan", hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                                         hdbscan_min_samples=hdbscan_min_samples)
            clusters = clusterer.fit_predict(distance_matrix)

            # Calculate the clustering metrics
            share_outliers = np.sum(clusters == -1) / len(clusters)
            buildings_per_cluster = pd.Series(clusters[clusters != -1]).value_counts()
            cluster_size_cv = float(buildings_per_cluster.std() / buildings_per_cluster.mean())

            # Print cluster metrics evaluation
            print(f"Coefficient of variation of cluster size: {cluster_size_cv:.2f}, "
                  f"Share of outliers: {share_outliers:.2f}, "
                  f"min cluster size: {hdbscan_min_cluster_size}, min_samples: {hdbscan_min_samples}")

            if share_outliers < lowest_share_outliers:
                print(f"New lowest share of outliers: {share_outliers:.2f}")
                lowest_share_outliers = share_outliers

            if cluster_size_cv < lowest_cluster_size_cv:
                print(f"New lowest coefficient of variation of cluster size: {cluster_size_cv:.2f}")
                lowest_cluster_size_cv = cluster_size_cv

            # Select the best cluster hyperparameters based on the solution's distance to the utopian point in the
            # hyperparameter space - a concept commonly used in compromise programming
            # (first described by Yu (1973), DOI: 10.1287/mnsc.19.8.936)
            this_utopian_point_distance = math.sqrt((share_outliers - lowest_share_outliers) ** 2 +
                                                    (cluster_size_cv - lowest_cluster_size_cv) ** 2)
            best_utopian_point_distance = math.sqrt((best_share_outliers - lowest_share_outliers) ** 2 +
                                                    (best_cluster_size_cv - lowest_cluster_size_cv) ** 2)
            if this_utopian_point_distance < best_utopian_point_distance:
                print(f"New best clusters assigned (utopian point distance: {this_utopian_point_distance:.2f} < "
                      f"{best_utopian_point_distance:.2f})")
                best_share_outliers = share_outliers
                best_cluster_size_cv = cluster_size_cv
                best_min_cluster_size = hdbscan_min_cluster_size
                best_min_samples = hdbscan_min_samples
                best_clusters = clusters

    print(f"Best clusters -- coefficient of variation of cluster size: {best_cluster_size_cv:.2f}, "
          f"share of outliers: {best_share_outliers:.2f}, "
          f"min cluster size: {best_min_cluster_size}, min_samples: {best_min_samples}")

    return best_clusters


def define_clusterer(approach, **kwargs):
    """
    Define the clustering approach to use and create the corresponding clusterer object.
    """
    # Default values
    dbscan_eps = DBSCAN_EPS
    dbscan_min_samples = DBSCAN_MIN_SAMPLES
    hdbscan_min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples = HDBSCAN_MIN_SAMPLES

    # Override default values with keyword arguments
    for key, value in kwargs.items():
        if key == 'dbscan_eps':
            dbscan_eps = value
        elif key == 'dbscan_min_samples':
            dbscan_min_samples = value
        elif key == 'hdbscan_min_cluster_size':
            hdbscan_min_cluster_size = value
        elif key == 'hdbscan_min_samples':
            hdbscan_min_samples = value
        else:
            raise Warning(f"Invalid keyword argument: {key}")

    # Define the clusterer object
    if approach == 'dbscan':
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='precomputed')
    elif approach == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples,
                                    metric='precomputed')
    else:
        raise ValueError(f"Invalid clustering approach: {approach}")

    return clusterer


def visualize_clusters(buildings_with_clusters, representative_districts=None):
    """
    Visualize the clusters on a map.
    """
    # Separate outliers (cluster == -1) from the main dataset
    outliers = buildings_with_clusters[buildings_with_clusters['cluster'] == -1]
    non_outliers = buildings_with_clusters[buildings_with_clusters['cluster'] != -1]

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the non-outliers (buildings with clusters)
    non_outliers.plot(column='cluster', cmap='Set3', legend=True, ax=ax)

    # Plot the outliers with black outline and white fill
    outliers.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5, marker='o')

    # Plot the representative districts with black outlines
    if representative_districts is not None:
        representative_districts.plot(ax=ax, color='black', edgecolor='red', linewidth=2)

    # Set the title
    ax.set_title('Representative Districts')

    # Show the plot
    plt.show()


def select_representative_districts():
    """
    Select a representative district from each cluster based on the optimal number of clusters using silhouette score.

    :return:
    """
    # Load the clustered building data
    buildings_with_clusters = gpd.read_file(CLUSTERED_BUILDING_DATA_PATH)

    # Step 1: Group by 'cluster' and 'LU_DESC' to get counts of land-use types per cluster
    cluster_landuse_counts = buildings_with_clusters[buildings_with_clusters['cluster']
                                                     != -1].groupby(['cluster', 'LU_DESC']).size().unstack(fill_value=0)

    # Step 2: Calculate the percentage of each land-use type within each cluster
    cluster_landuse_percentage = cluster_landuse_counts.div(cluster_landuse_counts.sum(axis=1), axis=0)
    cluster_landuse_array = cluster_landuse_percentage.fillna(0).to_numpy()  # Ensure no NaN values

    # Step 3: Dynamically select the number of clusters based on silhouette score
    best_num_clusters, best_kmedoids_model = select_optimal_number_of_clusters(cluster_landuse_array)

    # Step 4: Identify the representative districts
    medoid_ids = best_kmedoids_model.medoid_indices_
    representative_districts = buildings_with_clusters[buildings_with_clusters['cluster'].isin(medoid_ids)]

    # Save the representative districts to a Shape-file
    save_as_shapefile(representative_districts, medoid_ids)

    # Visualization
    visualize_clusters(buildings_with_clusters, representative_districts)


def select_optimal_number_of_clusters(cluster_landuse_array):
    """
    Select the optimal number of clusters based on silhouette score.

    :param cluster_landuse_array: Numpy array with percentage distributions of land-use types per cluster
    :return: Optimal number of clusters and the corresponding k-medoids model
    """
    best_num_clusters = MIN_LAND_USE_CLUSTERS  # Start with 2 clusters
    best_silhouette_score = -1  # Initialize to a low value
    best_kmedoids_model = None

    # Loop over different numbers of clusters and calculate silhouette scores
    for num_clusters in range(MIN_LAND_USE_CLUSTERS, MAX_LAND_USE_CLUSTERS + 1):
        kmedoids = KMedoids(n_clusters=num_clusters, method='fasterpam', metric=LAND_USE_DISTANCE_METRIC,
                            random_state=42)
        district_fit = kmedoids.fit(cluster_landuse_array)

        # Calculate silhouette score (ignoring outliers or -1 values)
        labels = district_fit.labels_
        score = silhouette_score(cluster_landuse_array, labels, metric=LAND_USE_DISTANCE_METRIC)

        print(f"Number of clusters: {num_clusters}, Silhouette score: {score}")

        # Keep track of the best model based on the silhouette score
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_num_clusters = num_clusters
            best_kmedoids_model = district_fit

    print(f"Best number of clusters: {best_num_clusters}, with silhouette score: {best_silhouette_score}")
    return best_num_clusters, best_kmedoids_model


def save_as_shapefile(all_representative_districts, medoid_ids):
    """
    Save the representative districts to a Shape-file.

    :param all_representative_districts: GeoDataFrame of representative districts
    :param medoid_ids: List of k-medoid cluster medoids' indices
    :return:
    """
    # Clear the directory if it already exists
    if os.path.exists(REPRESENTATIVE_DISTRICT_DIRECTORY):
        shutil.rmtree(REPRESENTATIVE_DISTRICT_DIRECTORY)

    for i, medoid_id in enumerate(medoid_ids):
        representative_district = all_representative_districts[all_representative_districts['cluster'] == medoid_id]

        # Create a unique subdirectory for each district
        district_dir = os.path.join(REPRESENTATIVE_DISTRICT_DIRECTORY, f"district_{i+1}")
        if not os.path.exists(district_dir):
            os.makedirs(district_dir)

        # Save the shapefile in the district subdirectory
        file_path = os.path.join(district_dir, f"zone.shp")
        try:
            representative_district.to_file(file_path, driver='ESRI Shapefile')
        except Exception as e:
            print(f"Error saving shapefile for district {i+1}: {e}")


# Entry point of the script
if __name__ == '__main__':
    create_geographical_clusters()
    select_representative_districts()
