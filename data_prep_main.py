import geopandas as gpd

from constants import BUILDING_DATA_PATH, LAND_PLOT_DATA_PATH, LAND_PLOT_ATTRIBUTES, BUILDING_PLOT_DATA_PATH


def clean_data():
    """
    Clean the building and land plot data by:
    1. Importing the data
    2. Parsing the HTML data in the land plot data
    3. Matching each building to a land plot
    4. Saving the cleaned data
    """
    # Import the data
    buildings = gpd.read_file(BUILDING_DATA_PATH, columns=['geometry'])
    land_plots = gpd.read_file(LAND_PLOT_DATA_PATH)

    # Parse html data in land_plots to more pythonic data
    for attr in LAND_PLOT_ATTRIBUTES:
        land_plots[attr] = land_plots.Description.apply(fast_extract_attributes, attribute_name=attr)

    # Clean-up land_plot dataframe - delete the original 'Description' column
    land_plots = land_plots.drop(columns=['Description'])

    # Ensure the data is in the same CRS (EPSG:3414, Mercator projection for Singapore)
    buildings = buildings.to_crs(epsg=3414)
    land_plots = land_plots.to_crs(epsg=3414)

    # Determine which land plot each building belongs to
    buildings_with_plot = gpd.sjoin(buildings, land_plots, how="left", predicate="within")

    # Drop all buildings that could not be matched to a land plot
    buildings_with_plot = buildings_with_plot.dropna(subset=['index_right'])
    buildings_with_plot = buildings_with_plot.drop(columns=['index_right'])

    # Save the cleaned data
    buildings_with_plot.to_file(BUILDING_PLOT_DATA_PATH, driver='GeoJSON')


def fast_extract_attributes(html_string, attribute_name='LU_DESC'):
    """
    Fast extraction of the LU_DESC attribute from the HTML string.

    :param html_string: A string containing the HTML attributes
    :param attribute_name: The name of the attribute to extract
    :return: The value of LU_DESC or None if not found
    """
    start_token = f"<th>{attribute_name}</th>"

    start_index = html_string.find(start_token)

    if start_index == -1:
        return None  # 'arrtibute_name' not found

    # Move index to start of the value after the start token
    start_index = start_index + len(start_token)

    # Look for the opening <td> tag
    td_start = html_string.find("<td>", start_index)

    if td_start == -1:
        return None  # <td> tag not found

    # Move index to the start of the actual value
    td_start = td_start + len("<td>")

    # Find the closing </td> tag
    td_end = html_string.find("</td>", td_start)

    if td_end == -1:
        return None  # Closing </td> tag not found

    # Extract the value of LU_DESC
    attribute_value = html_string[td_start:td_end].strip()

    return attribute_value


# Entry point of the script
if __name__ == '__main__':
    clean_data()

