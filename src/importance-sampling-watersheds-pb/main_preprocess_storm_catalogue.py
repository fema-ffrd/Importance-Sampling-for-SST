#region Libraries

#%%
import os

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from modules.preprocess_storm_catalogue import preprocess_storm_catalogue

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Set working folder
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py\Trinity')

    #%% Set location of folder with precipitation nc files and data name
    # folder_storms = r'D:\FEMA Innovations\SO3.1\Py\temp_storms'
    # folder_storms = r'D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\events'
    # nc_data_name = 'APCP_surface'
    folder_storms = r'D:\FEMA Innovations\SO3.1\Py\Trinity\nc'
    nc_data_name = 'band_data'
    
    #%% Preprocess the nc files, save storm catalogue to "path_storm"
    path_storm = 'storm_catalogue_trinity'
    preprocess_storm_catalogue(folder_storms, nc_data_name, path_storm)

#endregion -----------------------------------------------------------------------------------------
