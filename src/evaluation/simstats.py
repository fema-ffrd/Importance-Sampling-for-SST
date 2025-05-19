import numpy as np

def print_sim_stats(df_prob, multiplier=1):
    n_sim = df_prob.shape[0]
    n_sim_intersect = df_prob.loc[lambda _: _.intersected == 1].shape[0]
    rate_success = n_sim_intersect/n_sim*100

    prob_total = df_prob.prob.sum()
    prob_intersected = df_prob.loc[lambda _: _.intersected == 1].prob.sum()

    df_prob = \
    (df_prob
        .assign(x_px = lambda _: _.depth * _.prob)
    )
    mean = df_prob.x_px.sum()
    df_prob = \
    (df_prob
        .assign(x_mx_px = lambda _: ((_.depth - mean)**2) * _.prob)
    )
    std = np.sqrt(df_prob.x_mx_px.sum())
    standard_error = std/np.sqrt(n_sim)

    depth_weighted = df_prob.depth * df_prob.weight
    mean_estimate = np.mean(depth_weighted)
    std_estimate = np.std(depth_weighted, ddof=1) # Sample std dev of h(x)*w(x)
    standard_error_estimate = std_estimate / np.sqrt(n_sim)

    print(
        f'Intersected: {n_sim_intersect} out of {n_sim} ({rate_success:.2f}%)\n'
        + f'Total Weights: Total {prob_total: .2f}, Intersected: {prob_intersected:.2f}\n'
        + f'Depth: {mean*multiplier:.2f} ± {standard_error*multiplier:.2f}\n'
        + f'Depth Estimate: {mean_estimate*multiplier:.2f} ± {standard_error_estimate*multiplier:.2f}'
    )