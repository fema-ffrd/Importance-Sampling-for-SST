#region Libraries

#%%
import os

#%%
from modules.preprocess_storm_catalogue import preprocess_storm_catalogue

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%%
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py')

    #%%
    # folder_storms = r'D:\FEMA Innovations\SO3.1\Py\temp_storms'
    folder_storms = r'D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\events'
    nc_data_name = 'APCP_surface'
    
    #%%
    preprocess_storm_catalogue(folder_storms, nc_data_name, 'temp_storm_catalogue_2')

#endregion -----------------------------------------------------------------------------------------
