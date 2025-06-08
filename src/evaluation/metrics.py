#region Libraries

#%%
from typing import Literal

import numpy as np
import pandas as pd

from scipy import stats

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
def get_aep_rmse_iter(df_aep_mc_0: pd.DataFrame, df_aep_iter: pd.DataFrame) -> tuple[list, dict, str]:
    v_rmse = []
    for iter in df_aep_iter.iter.unique():
        _rmse = get_aep_rmse(df_aep_mc_0, df_aep_iter[lambda _: _.iter == iter])
    
        v_rmse.append(_rmse)
    
    d_rmse = dict(
        rmse_min = float(np.min(v_rmse)),
        rmse_max = float(np.max(v_rmse)),
        rmse_mean = float(np.mean(v_rmse)),
        rmse_median = float(np.median(v_rmse)),
        rmse_std = float(np.std(v_rmse, ddof=1)),
        rmse_se = float(stats.sem(v_rmse)), # standard error
        rmse_me = float(stats.t.ppf((1+0.95)/2, len(v_rmse)-1) * stats.sem(v_rmse)) # margin of error (95% CI)
    )
    
    rmse_iter = f"{round(d_rmse.get('rmse_mean'), 3)} ± {round(d_rmse.get('rmse_me'), 3)}"
    
    return v_rmse, d_rmse, rmse_iter

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
