#region Libraries

#%%
import numpy as np
import pandas as pd

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
    df_prob_mc = pd.DataFrame(dict(
        depth = depths,
        prob = probs
    ))

    # Exceedence probability
    df_prob_mc = \
    (df_prob_mc
        .sort_values('depth', ascending=False)
        .assign(prob_exceed_pds = lambda _: _.prob.cumsum())
        .assign(prob_exceed = lambda _: 1 - np.exp(-lambda_rate * _.prob_exceed_pds))
        .assign(return_period = lambda _: 1/_.prob_exceed)
    )

    return df_prob_mc

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
