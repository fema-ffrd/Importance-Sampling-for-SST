#region Libraries

#%%
from typing import Literal

import numpy as np
import pandas as pd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
#%%
#TODO
def get_aep_rmse(df_aep_1: pd.DataFrame, df_aep_2: pd.DataFrame) -> float:
    mse = \
    (df_aep_1
        [['return_period', 'depth']]
        .rename(columns={'depth': 'depth_1'})
        .merge(df_aep_2[['return_period', 'depth']].rename(columns={'depth': 'depth_2'}), on='return_period')
        .dropna()
        .assign(diff2 = lambda _: (_.depth_1 - _.depth_2)**2)
        .diff2
        .mean()
    )
    rmse = float(np.sqrt(mse))

    return rmse

#%%
#TODO
def get_aep_cost_effectiveness(df_aep_mc_0: pd.DataFrame, df_aep_mc: pd.DataFrame, df_aep_is: pd.DataFrame, cost_1: float, cost_2: float, normalize: Literal['none', 'mean', 'std'] = 'mean') -> float:
    v_return_period = \
    (df_aep_mc_0
        .merge(df_aep_mc, on='return_period')
        .merge(df_aep_is, on='return_period')
        .dropna()
        ['return_period'] 
    )
    
    df_aep_mc_0 = df_aep_mc_0.loc[lambda _: _.return_period.isin(v_return_period)]
    df_aep_mc = df_aep_mc.loc[lambda _: _.return_period.isin(v_return_period)]
    df_aep_is = df_aep_is.loc[lambda _: _.return_period.isin(v_return_period)]

    rmse_1 = get_aep_rmse(df_aep_mc_0, df_aep_mc)
    rmse_2 = get_aep_rmse(df_aep_mc_0, df_aep_is)

    ce = (rmse_1 - rmse_2)/(cost_1 - cost_2)

    match normalize:
        case 'none':
            factor_normalize = 1
        case 'mean':
            factor_normalize = df_aep_mc_0.depth.mean()
        case 'std':
            factor_normalize = df_aep_mc_0.depth.std()

    ce = float(ce/factor_normalize)

    return ce

#endregion -----------------------------------------------------------------------------------------
