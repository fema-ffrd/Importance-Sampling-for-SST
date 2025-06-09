#region Libraries

#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Create probability dataframe from depths (sorted) and weights (sorted)
def get_return_period_langbein(df_depths: pd.DataFrame, lambda_rate=10) -> pd.DataFrame:
    '''Generate frequency distribution curve dataframe

    Args:
        df_depths (pd.DataFrame): Dataframe of depths and probabilities.
        lambda_rate (float): Rate of flood events (=m/N where m is the number of events in the partial duration series and N is the number of years). Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with inverse sorted depths and corresponding probabilities, exceedence probabilities, and return periods.
    '''
    # Table of depths and probabilities
    df_prob = pd.DataFrame(dict(
        depth = df_depths.depth,
        prob = df_depths.prob,
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
def get_aep_depths(df_prob: pd.DataFrame, return_period: list|np.ndarray|pd.Series = [2, 2.5, 4, 5, 6.67, 10, 12.5, 20, 25, 33.33, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 2000, 5000], min_events_per_year = None) -> pd.DataFrame:
    v_depths =  interp1d(df_prob.return_period, df_prob.depth, bounds_error=False)(return_period)

    df_aep = pd.DataFrame(dict(return_period = return_period))

    df_aep = \
    (df_aep
        .assign(prob_exceed = lambda _: 1/_.return_period)
        .assign(depth = v_depths)
    )

    if min_events_per_year is not None:
        _return_period_max = df_prob.shape[0]/min_events_per_year
        # return_period = [_ for _ in return_period if _ <= _return_period_max]

        df_aep = \
        (df_aep
            .assign(depth = lambda _: np.where(_.return_period <= _return_period_max, _.depth, np.nan))
        )

    return df_aep

#%%
#TODO
def get_aep_uncertainty(df_aep_iter: pd.DataFrame, group_col = 'return_period') -> pd.DataFrame:
    df_aep_iter_summary = \
    (df_aep_iter
        .groupby(group_col)
        .agg(
            mean = ('depth', 'mean'),
            median = ('depth', 'median'),
            min = ('depth', 'min'),
            max = ('depth', 'max'),
        )
        .reset_index()
        .melt(id_vars=group_col, var_name='type_val', value_name='depth')
    )
    
    return df_aep_iter_summary

#endregion -----------------------------------------------------------------------------------------
