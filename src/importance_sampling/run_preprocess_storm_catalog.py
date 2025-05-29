#region Libraries

#%%
import os
import pathlib

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_preprocessor import preprocess_storm_catalog

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][0]
    
    #%% Working folder
    os.chdir(rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}')
    cwd = pathlib.Path.cwd()
    
    #%% Preprocess the nc files, save storm catalogue to "path_storm"
    preprocess_storm_catalog(cwd/'nc', 'band_data')

#endregion -----------------------------------------------------------------------------------------
