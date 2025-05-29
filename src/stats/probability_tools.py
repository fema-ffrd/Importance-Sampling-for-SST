#region Libraries

#%%
import numpy as np
import pandas as pd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Compute probabilities
def get_prob(df_depths: pd.DataFrame, greater_than: list = None,  greater_than_incl = True, less_than: list = None,less_than_incl = True) -> pd.DataFrame:
    '''Calculates the probability of depth being greater than or less than a value or between two values.

    Args:
        df_depths (pd.DataFrame): Dataframe of probabilities from 'compute_depths'.
        greater_than: The threshold greater than which to calculate the probability.
        greater_than_incl: If True, condition is depth >= greater_than.
                           If False, condition is depth > greater_than.
        less_than: The threshold less than which to calculate the probability.
        less_than_incl: If True, condition is depth <= less_than.
                        If False, condition is depth < less_than.
    '''
    v_depth = df_depths.depth
    # v_weight = df_depths.weight
    v_prob = df_depths.prob
    if greater_than is not None:
        if greater_than_incl:
            indicator = (v_depth >= greater_than)
        else:
            indicator = (v_depth > greater_than)
        # prob_greater_than = np.mean(indicator * v_weight)
        prob_greater_than = np.sum(indicator * v_prob)
    if less_than is not None:
        if less_than_incl:
            indicator = (v_depth <= less_than)
        else:
            indicator = (v_depth < less_than)
        # prob_less_than = np.mean(indicator * v_weight)
        prob_less_than = np.sum(indicator * v_prob)

    if greater_than is not None and less_than is not None:
        prob = prob_less_than - prob_greater_than
    elif greater_than is not None:
        prob = prob_greater_than
    elif less_than is not None:
        prob = prob_less_than
    else:
        prob = 1

    return prob

#endregion -----------------------------------------------------------------------------------------
