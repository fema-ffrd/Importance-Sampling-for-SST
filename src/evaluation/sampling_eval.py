#region Libraries

#%%
import numpy as np
import pandas as pd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Print simulation statistics
def print_sim_stats(df_depths: pd.DataFrame, multiplier: float=1) -> None:
    '''Print various simulation statistics such as proportion of samples with non-zero depth, and mean and standard error of depths.

    Args:
        df_depths (pd.DataFrame): Dataframe of probabilities from 'compute_depths'.
        multiplier (float, optional): A multiplier for deth values. Defaults to 1.
    '''
    n_sim = df_depths.shape[0]
    n_sim_intersect = df_depths.loc[lambda _: _.intersected == 1].shape[0]
    rate_success = n_sim_intersect/n_sim*100

    prob_total = df_depths.prob.sum()
    prob_intersected = df_depths.loc[lambda _: _.intersected == 1].prob.sum()

    weight_sum = df_depths.weight.sum()

    weight_sum_to_n_sim = weight_sum/n_sim

    ESS = 1 / ((df_depths.prob)**2).sum()
    # ESS = (df_depths.weight.sum())**2/(df_depths.weight**2).sum()

    df_depths = \
    (df_depths
        .assign(x_px = lambda _: _.depth * _.prob)
    )
    mean_estimate_n = df_depths.x_px.sum()
    df_depths = \
    (df_depths
        .assign(x_mx_px = lambda _: ((_.depth - mean_estimate_n)**2) * _.prob)
    )
    std_estimate_n = np.sqrt(df_depths.x_mx_px.sum())
    standard_error_estimate_n = std_estimate_n/np.sqrt(n_sim)

    depth_weighted_n = df_depths.depth * df_depths.weight
    mean_estimate_n = np.sum(depth_weighted_n)/df_depths.weight.sum()
    standard_error_estimate_n = np.sqrt(np.sum(((df_depths.depth - mean_estimate_n)**2 * df_depths.weight)))/df_depths.weight.sum()

    depth_weighted_n = df_depths.depth * df_depths.prob
    mean_estimate_n = np.sum(depth_weighted_n)
    standard_error_estimate_n = np.sqrt(np.sum((df_depths.depth - mean_estimate_n)**2 * df_depths.prob))

    depth_weighted_n = df_depths.depth * df_depths.weight
    mean_estimate_n = np.sum(depth_weighted_n)/df_depths.weight.sum()
    _var_term = ((df_depths.depth - mean_estimate_n)*df_depths.weight)**2
    var_estimate_n = _var_term.sum()/(df_depths.weight.sum())**2 # #*n_sim/(n_sim-1)
    standard_error_estimate_n = np.sqrt(var_estimate_n)

    depth_weighted_n = df_depths.depth * df_depths.prob
    mean_estimate_n = (df_depths.depth * df_depths.prob).sum()
    var_estimate_n = (df_depths.prob * (df_depths.depth - mean_estimate_n)**2).sum()
    standard_error_estimate_n = np.sqrt(var_estimate_n)/np.sqrt(n_sim)
    standard_error_estimate_n = np.sqrt(var_estimate_n)/np.sqrt(ESS)

    depth_weighted_un = df_depths.depth * df_depths.weight
    mean_estimate_un = np.mean(depth_weighted_un)
    std_estimate_un = np.std(depth_weighted_un, ddof=1) # Sample std dev of h(x)*w(x)
    standard_error_estimate_un = std_estimate_un / np.sqrt(n_sim)

    print(
        f'Intersected: {n_sim_intersect} out of {n_sim} ({rate_success:.2f}%)\n'
        + f'Total Weights: Total {prob_total: .2f}, Intersected: {prob_intersected:.2f}\n'
        + f'Weight sum/N simulations: {weight_sum_to_n_sim}\n'
        + f'ESS: {ESS:.0f}\n'
        + f'Depth Estimate (self-normalized): {mean_estimate_n*multiplier:.2f} ± {standard_error_estimate_n*multiplier:.2f}\n'
        + f'Depth Estimate (non-normalized): {mean_estimate_un*multiplier:.2f} ± {standard_error_estimate_un*multiplier:.2f}\n'
    )

#endregion -----------------------------------------------------------------------------------------
