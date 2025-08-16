# -*- coding: utf-8 -*-


# Script Description ##########################################################
"""
Convert the NSI structures table to a format that can be used by Hazus

NOTE: This script will take the output from the GET NSI STRUCTURES DATA tool 
in Boathouse and convert it to a CSV that is readable into the HAZUS script
(dmg2_DepthDamageCalculations.py). This should also work with data downloaded 
driectly from the NSI website, assuming it is in geopackage format. A couple 
of notes:
- Foundation Type is converted based on the logic Jill used in Indian Creek
- Occupancy Type is adjusted to remove any text after the hyphen

"""

# Imports #####################################################################
import geopandas as gpd
import pandas as pd

# Function to calculate f_foundtyp
def calculate_f_foundtyp(f_found_type, f_occtype):
    """
    Calculate the foundation type based on the found_type and occtype.

    NSI foundation_codes (f_found_type):
        C = Crawl, B = Basement, S = Slab, P = Pier, I = Pile, F = Fill, W = Solid Wall
    HAZUS foundation_codes: ?

    
    Parameters:
        f_found_type (str): The found_type value.
        f_occtype (str): The occtype value.
        
    Returns:
        int: The calculated foundation type.
    """

    if f_found_type != ' ':
        if f_found_type == 'I':
            return 1
        elif f_found_type == 'P':
            return 2
        elif f_found_type == 'W':
            return 3
        elif f_found_type == 'B':
            return 4
        elif f_found_type == 'C':
            return 5
        elif f_found_type == 'F':
            return 6
        elif f_found_type == 'S':
            return 7
    elif f_found_type == ' ':
        if f_occtype in ['RES1', 'RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F']:
            return 4
        elif f_occtype in ['GOV2', 'IND5']:
            return 5
        else:
            return 7
    else:
        pass

# Remove text after a hyphen
def drop_text_after_hyphen(field): 
    return field.split('-')[0]

# Read in the source NSI geopackage
def read_geopackage_to_df(gpkg, layer):
    # read in geopackage
    gdf = gpd.read_file(gpkg, layer=layer)

    # convert to dataframe
    df = pd.DataFrame(gdf)

    return(df)


def main_nsi_to_hazus(nsi_gpkg, nsi_layer, output_csv_file_path):

    df = read_geopackage_to_df(nsi_gpkg, nsi_layer)

    df['occtype'] = df.apply(lambda row: drop_text_after_hyphen(row['occtype']), axis=1)

    # Apply the function to create a new column 'calculated_f_foundtyp'
    df['FoundationType'] = df.apply(lambda row: calculate_f_foundtyp(row['found_type'], row['occtype']), axis=1)

    # Rename multiple columns using a dictionary
    column_mapping = {"bid": "id",
                    #'med_yr_blt': 'YEARBUILT', 
                    #'sqft': 'Area',
                    'found_type': 'FoundationType',
                    'val_struct': 'Cost',
                    'val_cont': "ContentCost",
                    'found_ht': 'FirstFloorHt',
                    #'num_story': 'NumStories',
                    'occtype': 'Occ',
                    'x': 'Longitude',
                    'y': 'Latitude'
                    }
    df.rename(columns=column_mapping, inplace=True)
    # I think I saw that Jill said num stories wasn't always accurate. She had a separate calculation that I did not apply here.

    #columns_to_keep = ['fd_id', 'Occ', 'NumStories', 'FoundationType', 'YEARBUILT', 'Area',"ContentCost", 'Cost', 'FirstFloorHt', 'Latitude', 'Longitude']
    columns_to_keep = ['fd_id', 'Occ', 'FoundationType',"ContentCost", 'Cost', 'FirstFloorHt', 'Latitude', 'Longitude']

    # Filter DataFrame to keep only the specified columns
    df = df[columns_to_keep]

    df.to_csv(output_csv_file_path, index=False)
    #return(df)


'''

# Specify the path to your CSV file
input_csv_path = 'data/0_source/Niver_South_Structures_raw.csv'
output_csv_file_path = 'data/0_source/Niver_South_Structures.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_path)'''

# Script ######################################################################

if __name__ == "__main__": 
    #currDir = os.path.dirname(os.path.realpath(__file__))
    main_nsi_to_hazus(nsi_gpkg = '/workspaces/Importance-Sampling-for-SST/data/0_source/Denton/purdue_si.gpkg',
                      nsi_layer = 'SI_HUC08',
                      output_csv_file_path = '/workspaces/Importance-Sampling-for-SST/data/1_interim/Denton/WCLC_structures_Purdue.csv')