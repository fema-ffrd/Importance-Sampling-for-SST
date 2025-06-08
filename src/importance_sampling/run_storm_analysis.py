#region Libraries

#%%
import os
import pathlib
import platform

import pandas as pd

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.utils_spatial.raster_stats import get_storm_stats

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Select watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]
    folder_watershed = rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}'

    #%% Working folder
    os.chdir(folder_watershed)
    cwd = pathlib.Path.cwd()
    
    #%% Run storm analysis
    get_storm_stats(folder_watershed)

#endregion -----------------------------------------------------------------------------------------
