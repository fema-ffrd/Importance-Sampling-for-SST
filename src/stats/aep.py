#region Libraries

#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Create probability dataframe from depths (sorted) and weights (sorted)
def get_return_period_langbein(depths: list|np.ndarray|pd.Series, probs: list|np.ndarray|pd.Series, lambda_rate=10) -> pd.DataFrame:
    '''Generate frequency distribution curve datafra.e

    Args:
        depths (list | np.ndarray | pd.Series): Vector of depths.
        probs (list | np.ndarray | pd.Series): Vector of probabilities.
        lambda_rate (float): Rate of flood events (=m/N where m is the number of events in the partial duration series and N is the number of years). Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with inverse sorted depths and corresponding probabilities, exceedence probabilities, and return periods.
    '''
    # Table of depths and probabilities
    df_prob = pd.DataFrame(dict(
        depth = depths,
        prob = probs
    ))

    # Exceedence probability
    df_prob = \
    (df_prob
        .sort_values('depth', ascending=False)
        .assign(prob_exceed_pds = lambda _: _.prob.cumsum())
        .assign(prob_exceed = lambda _: 1 - np.exp(-lambda_rate * _.prob_exceed_pds))
        .assign(return_period = lambda _: 1/_.prob_exceed)
    )

    return df_prob

#%%
#TODO
def get_aep_depths(df_prob: pd.DataFrame, return_period: list|np.ndarray|pd.Series = [2, 2.5, 4, 5, 6.67, 10, 12.5, 20, 25, 33.33, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 2000]) -> pd.DataFrame:
    v_depths =  interp1d(df_prob.return_period, df_prob.depth, bounds_error=False)(return_period)

    df_aep = pd.DataFrame(dict(return_period = return_period))

    df_aep = \
    (df_aep
        .assign(prob_exceed = lambda _: 1/_.return_period)
        .assign(depth = v_depths)
    )

    return df_aep

#%%
#TODO
def get_aep_uncertainty(df_prob: pd.DataFrame, return_period: list|np.ndarray|pd.Series = [2, 2.5, 4, 5, 6.67, 10, 12.5, 20, 25, 33.33, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 2000]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_aep = pd.DataFrame()
    for i in df_prob.iter.unique():
        _df_aep = get_aep_depths(df_prob.loc[lambda _: _.iter == i], return_period)
        _df_aep = _df_aep.assign(iter = i)
        df_aep = pd.concat([df_aep, _df_aep])

    df_aep_summary = \
    (df_aep
        .groupby('return_period')
        .agg(
            mean = ('depth', 'mean'),
            median = ('depth', 'median'),
            min = ('depth', 'min'),
            max = ('depth', 'max'),
        )
        .reset_index()
        .melt(id_vars='return_period', var_name='type', value_name='depth')
    )
    
    return df_aep, df_aep_summary

#endregion -----------------------------------------------------------------------------------------
#region Archive

# #%% Create probability dataframe from depths (sorted) and weights (sorted)
# def get_df_freq_curve(depths: list|np.ndarray|pd.Series, probs: list|np.ndarray|pd.Series) -> pd.DataFrame:
#     '''Generate frequency distribution curve datafra.e

#     Args:
#         depths (list | np.ndarray | pd.Series): Vector of depths.
#         probs (list | np.ndarray | pd.Series): Vector of probabilities.

#     Returns:
#         pd.DataFrame: Dataframe with inverse sorted depths and corresponding probabilities, exceedence probabilities, and return periods.
#     '''
#     # Table of depths and probabilities
#     df_prob_mc = pd.DataFrame(dict(
#         depth = depths,
#         prob = probs
#     ))

#     # Exceedence probability
#     df_prob_mc = \
#     (df_prob_mc
#         .sort_values('depth', ascending=False)
#         .assign(prob_exceed = lambda _: _.prob.cumsum())
#         .assign(return_period = lambda _: 1/_.prob_exceed)
#     )

#     return df_prob_mc

#endregion -----------------------------------------------------------------------------------------
