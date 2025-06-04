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
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]

    #%% Read parameters
    df_dist_params = pd.read_csv(r'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\src\importance_sampling\distribution_params.csv')

    #%% Working folder
    os.chdir(rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}')
    cwd = pathlib.Path.cwd()
    
    #%%
    df_storm_stats = get_storm_stats(cwd)

    #%%
    df_storm_stats.to_pickle(cwd/'pickle'/'df_storm_stats.pkl')

    #%%
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.pairplot(df_storm_stats.drop(columns=['x', 'y']))
    # plt.show()

#endregion -----------------------------------------------------------------------------------------
