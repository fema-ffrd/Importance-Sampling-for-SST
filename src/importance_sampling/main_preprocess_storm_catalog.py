#region Libraries

#%%
import os

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from modules.preprocess_storm_catalog import preprocess_storm_catalog

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Set working folder
    # pls UPDATE this: folder to save outputs
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py\Kanahwa')

    #%% Set location of folder with precipitation nc files and data name
    # pls UPDATE this: folder with input storms and band data for nc file
    # folder_storms = r'D:\FEMA Innovations\SO3.1\Py\temp_storms'
    # folder_storms = r'D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\events'
    # nc_data_name = 'APCP_surface'
    folder_storms = r'D:\FEMA Innovations\SO3.1\Py\Kanahwa\nc'
    nc_data_name = 'band_data'

    #%% Preprocess the nc files, save storm catalogue to "path_storm"
    # pls UPDATE this: name for catalogue folder
    preprocess_storm_catalog(folder_storms, nc_data_name)

#endregion -----------------------------------------------------------------------------------------
