""" This script is used to convert the representative districts to the right format for CEA. """

import os
import shutil
import fiona
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.validation import explain_validity


from constants import REPRESENTATIVE_DISTRICT_DIRECTORY, OVERLAP_THRESHOLD, PLOT_MATCHED_BUILDINGS, CEA_FILES_DIRECTORY


def load_representative_districts():
    """Load the representative districts. """
    # Fetch project directory
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Fetch the path to the representative districts
    representative_districts_path = os.path.join(project_directory, REPRESENTATIVE_DISTRICT_DIRECTORY)

    # Fetch the individual representative district directories
    district_dirs = [f.path for f in os.scandir(representative_districts_path) if f.is_dir()]
    representative_districts = {}

    # Load the representative districts
    for district_dir in district_dirs:
        district_name = district_dir.split('/')[-1]
        representative_districts[district_name] = gpd.read_file(f'{district_dir}/zone.shp')

    return representative_districts


def fetch_osm_data(representative_districts):
    """Fetch OSM data for the buildings in the representative districts."""
    representative_districts_osm = {}

    # Iterate over the representative districts
    for district_name, district_buildings in representative_districts.items():
        # Ensure district buildings remain in EPSG:3414
        district_crs = district_buildings.crs

        # Reproject district buildings to WGS84 (EPSG:4326) for OSM querying
        district_buildings_wgs84 = district_buildings.to_crs(epsg=4326)

        # Calculate the convex hull of the buildings in the district (now in EPSG:4326)
        district_hull_wgs84 = district_buildings_wgs84.unary_union.convex_hull

        # Buffer the hull to avoid invalid geometries
        district_hull_wgs84 = district_hull_wgs84.buffer(0.00001)

        # Validate and fix the geometry of the hull
        district_hull_wgs84 = validate_and_fix_geometry(district_hull_wgs84)

        # Fetch the buildings in the district from OSM (since it's in EPSG:4326)
        osm_buildings = ox.features_from_polygon(district_hull_wgs84, tags={'building': True})

        # Reproject OSM buildings back to the original CRS (EPSG:3414)
        osm_buildings = osm_buildings.to_crs(district_crs)

        # Match the buildings with the buildings in the representative district (in EPSG:3414 now)
        matched_buildings = gpd.sjoin(district_buildings, osm_buildings, how='inner', predicate='intersects')

        # Filter out matched buildings based on overlap with original buildings
        filtered_buildings = filter_matched_buildings_by_overlap(district_buildings, matched_buildings)

        # Remove overlapping buildings
        buildings = remove_overlapping_buildings(filtered_buildings)

        # Plot the matched buildings
        if PLOT_MATCHED_BUILDINGS:
            plot_buildings(district_buildings, buildings)

        representative_districts_osm[district_name] = buildings

    return representative_districts_osm


def validate_and_fix_geometry(polygon):
    if not polygon.is_valid:
        print(f"Invalid geometry detected: {explain_validity(polygon)}")
        # Attempt to fix invalid geometry using buffer(0) trick
        polygon = polygon.buffer(0)
        if polygon.is_valid:
            print("Geometry fixed using buffer(0)")
        else:
            print("Unable to fix geometry")
    return polygon


def plot_buildings(building_set_1, building_set_2):
    # Plot original buildings in one color
    ax = building_set_1.plot(color='blue', alpha=0.5)

    # Plot matched OSM buildings in another color, on the same axes
    building_set_2.plot(color='red', alpha=0.5, ax=ax)

    # Create custom legend handles
    blue_patch = mpatches.Patch(color='blue', label='Original Buildings')
    red_patch = mpatches.Patch(color='red', label='Matched OSM Buildings')

    # Add legend manually
    plt.legend(handles=[blue_patch, red_patch])

    # Show the plot
    plt.show()


def filter_matched_buildings_by_overlap(district_buildings, matched_buildings, overlap_threshold=OVERLAP_THRESHOLD):
    """
    Filter out matched buildings based on overlap with original buildings.

    :param district_buildings: GeoDataFrame of original district buildings.
    :param matched_buildings: GeoDataFrame of matched OSM buildings.
    :param overlap_threshold: Minimum overlap percentage required to keep a building (e.g., 0.5 for 50% overlap).
    :return: GeoDataFrame of filtered matched buildings.
    """
    # Ensure geometries are aligned (in case CRS differs)
    district_buildings = district_buildings.to_crs(matched_buildings.crs)

    # Calculate intersection area between each matched and original building
    matched_buildings['intersection_area'] = matched_buildings.geometry.intersection(
        district_buildings.unary_union).area

    # Calculate the area of the matched buildings
    matched_buildings['matched_building_area'] = matched_buildings.geometry.area

    # Calculate the percentage overlap
    matched_buildings['overlap_percentage'] = matched_buildings['intersection_area'] / \
                                              matched_buildings['matched_building_area']

    # Filter buildings where the overlap percentage is greater than or equal to the threshold
    filtered_buildings = matched_buildings[matched_buildings['overlap_percentage'] >= overlap_threshold]

    # Drop intermediate columns if needed
    filtered_buildings = filtered_buildings.drop(columns=['intersection_area', 'matched_building_area',
                                                          'overlap_percentage'])

    return filtered_buildings


def remove_overlapping_buildings(buildings):
    """
    Remove overlapping polygons according to the rules specified:
    - 100% overlap: Keep the one with the most complete attributes.
    - Partial overlap: Keep the higher building, cutting out the overlap from the lower building.

    :param buildings: GeoDataFrame containing matched buildings with a 'height' attribute.
    :return: GeoDataFrame of filtered and non-overlapping matched buildings.
    """
    # Sort buildings by completeness of attributes and height (priority goes to more complete and higher buildings)
    buildings['num_attributes'] = buildings.drop(columns=['geometry']).count(axis=1)
    buildings = buildings.sort_values(by=['num_attributes'], ascending=[False])

    # Prepare a list to store non-overlapping polygons
    clean_buildings = gpd.GeoDataFrame(columns=buildings.columns, crs=buildings.crs)

    for idx, building in buildings.iterrows():
        # Check for overlap with existing buildings in final set
        overlaps = clean_buildings.geometry.apply(lambda geom: geom.intersects(building.geometry))

        if overlaps.any():
            overlapping_buildings = clean_buildings[overlaps]
            for _, overlapping_building in overlapping_buildings.iterrows():
                # Calculate the intersection
                intersection = building.geometry.intersection(overlapping_building.geometry)

                if intersection.area == building.geometry.area:
                    # 100% overlap: Keep only the one with more attributes
                    continue  # The more complete one is already first due to sorting

                elif intersection.area > 0 and 'height' in building.keys():
                    # Partial overlap: cut out the overlap from the lower building
                    if float(building['height']) > float(overlapping_building['height']):
                        # Subtract the overlap from the lower building's geometry
                        clean_buildings.loc[
                            overlapping_building.name, 'geometry'] = overlapping_building.geometry.difference(
                            building.geometry)
                    else:
                        # Subtract the overlap from the current building
                        building['geometry'] = building.geometry.difference(overlapping_building.geometry)

                elif intersection.area > 0 and 'height' not in building.keys():
                    # Partial overlap without height information: cut out the overlap from the building with fewer attributes
                    if building['num_attributes'] > overlapping_building['num_attributes']:
                        # Subtract the overlap from the lower building's geometry
                        clean_buildings.loc[
                            overlapping_building.name, 'geometry'] = overlapping_building.geometry.difference(
                            building.geometry)
                    else:
                        # Subtract the overlap from the current building
                        building['geometry'] = building.geometry.difference(overlapping_building.geometry)

        # Add the building to the final set if it has valid geometry left
        if not building.geometry.is_empty:
            clean_buildings = pd.concat([clean_buildings, pd.DataFrame([building])], ignore_index=True)

    # Drop the intermediate 'num_attributes' column before returning
    clean_buildings = clean_buildings.drop(columns=['num_attributes'])

    return clean_buildings


def prepare_cea_data(representative_districts_osm):
    """Convert OSM building attributes into CEA format."""

    cea_districts = {}

    for district_name, buildings in representative_districts_osm.items():
        # Initialize an empty dataframe to store CEA attributes
        cea_buildings = gpd.GeoDataFrame(columns=[
            'Name', 'height_ag', 'floors_ag', 'height_bg', 'floors_bg', 'descriptio',
            'house_name', 'house_no', 'street', 'postcode', 'city', 'country',
            'STANDARD', 'YEAR', '1ST_USE', '1ST_USE_R', '2ND_USE', '2ND_USE_R',
            '3RD_USE', '3RD_USE_R', 'REFERENCE'], crs=buildings.crs)

        building_data_list = []

        for idx, building in buildings.iterrows():
            # Extract attributes from OSM data
            building_data = {
                'geometry': building['geometry'],
                'Name': str(building['Name']) if 'Name' in building else f'Building_{idx}',
                'height_ag': float(building['height']) if 'height' in building else None,
                'floors_ag': int(building['building:levels']) if 'building:levels' in building else None,
                'height_bg': None,
                'floors_bg': None,  # Same as above
                'descriptio': str(building['description']) if 'description' in building else None,
                'house_name': str(building['addr:housename']) if 'addr:housename' in building else None,
                'house_no': str(building['addr:housenumber']) if 'addr:housenumber' in building else None,
                'street': str(building['addr:street']) if 'addr:street' in building else None,
                'postcode': int(building['addr:postcode']) if 'addr:postcode' in building else None,
                'city': str(building['addr:city']) if 'addr:city' in building else None,
                'country': str(building['addr:country']) if 'addr:country' in building else None,
                'STANDARD': 'STANDARD1',  # Default value
                'YEAR': int(building['start_date']) if 'start_date' in building else None,
                '1ST_USE': str(building['building']) if 'building' in building else None,
                '1ST_USE_R': 1,  # Assuming 100% for now, adjust later if needed
                '2ND_USE': None,
                '2ND_USE_R': 0,
                '3RD_USE': None,
                '3RD_USE_R': 0,
                'REFERENCE': 'OSM'  # Source of data
            }

            building_data_list.append(building_data)

        # Create a temporary DataFrame from the list
        temp_df = pd.DataFrame(building_data_list)

        # Convert it into a GeoDataFrame, ensuring the geometry column is properly handled
        cea_buildings = gpd.GeoDataFrame(temp_df, geometry='geometry', crs=buildings.crs)

        # Assign the processed buildings to the district
        cea_districts[district_name] = cea_buildings

    return cea_districts


def save_districts_to_files(representative_districts_cea, output_directory=CEA_FILES_DIRECTORY):
    """
    Save the representative district data to 'zone.shp' and 'typology.dbf' files.

    :param representative_districts_cea: Dictionary containing the processed CEA data for each district.
    :param output_directory: Directory where the files will be saved.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_output_directory = os.path.join(parent_dir, output_directory)
    if os.path.exists(full_output_directory):
        shutil.rmtree(full_output_directory)


    for district_name, cea_buildings in representative_districts_cea.items():
        # Create the output directory for each district
        district_output_dir = os.path.join(full_output_directory, district_name)
        os.makedirs(district_output_dir, exist_ok=True)

        # Define the fields for 'zone.shp' and 'typology.dbf'
        zone_fields = ['Name', 'height_ag', 'floors_ag', 'height_bg', 'floors_bg', 'descriptio',
                       'house_name', 'house_no', 'street', 'postcode', 'city', 'country']
        typology_fields = ['Name', 'STANDARD', 'YEAR', '1ST_USE', '1ST_USE_R',
                           '2ND_USE', '2ND_USE_R', '3RD_USE', '3RD_USE_R', 'REFERENCE']

        # Ensure we include the geometry for saving shapefiles
        zone_file_path = os.path.join(district_output_dir, 'zone.shp')
        cea_buildings_zone = cea_buildings[
            zone_fields + ['geometry']].copy()  # Ensure 'geometry' is included for GeoDataFrame
        cea_buildings_zone.to_file(zone_file_path)

        # Remove geometry for the typology DBF file since it doesn't need geometries
        typology_file_path = os.path.join(district_output_dir, 'typology.dbf')
        cea_buildings_typology = cea_buildings[typology_fields].copy()  # No geometry needed for DBF

        # Prepare a schema for the DBF file
        schema = {'properties': {col: 'str' for col in typology_fields}, 'geometry': 'None'}

        # Write the DBF file using Fiona
        with fiona.open(typology_file_path, mode='w', driver='ESRI Shapefile', schema=schema) as output:
            for _, row in cea_buildings_typology.iterrows():
                output.write({
                    'properties': row.to_dict(),
                    'geometry': None  # No geometry for the DBF file
                })

        print(f"Saved files for district '{district_name}' to {district_output_dir}")


def cea_data_prep_main():
    """Main function converting the representative districts to the right format for CEA. """
    # Load the representative districts
    representative_districts = load_representative_districts()

    # Fetch OSM data for the buildings in the representative districts
    representative_districts_osm = fetch_osm_data(representative_districts)

    # Extract and infer data relevant to cea from the representative districts and OSM data
    representative_districts_cea = prepare_cea_data(representative_districts_osm)

    # Save the CEA data to disk
    save_districts_to_files(representative_districts_cea)

    return None


if __name__ == '__main__':
    cea_data_prep_main()
