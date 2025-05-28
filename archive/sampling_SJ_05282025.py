from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import uniform
import geopandas as gpd
from pyDOE2 import lhs
from scipy.stats import uniform
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal

def sample_storm_centers(v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000) -> pd.DataFrame:
    '''Get storm center samples.

    Args:
        v_domain_stats (pd.Series): Stats of the transposition domain from 'get_sp_stats'.
        dist_x (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        dist_y (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        num_simulations (int, optional): Number of simulations. Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe of storm centers with columns 'x_sampled' (sampled x), 'y_sampled' (sampled y), 'prob' (probabilities that sum to 1), and 'weight' (weights for importance sampling, 1 for Monte Carlo sampling).
    '''
    if dist_x is None:
        method = 0

        dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
        dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)
    else:
        method = 1

    # Get storm centers and weights
    v_centroid_x = dist_x.rvs(num_simulations)
    v_centroid_y = dist_y.rvs(num_simulations)

    if method == 1:
        f_X_U = 1 / v_domain_stats.range_x
        f_Y_U = 1 / v_domain_stats.range_y
        f_X_TN = dist_x.pdf(v_centroid_x)
        f_Y_TN = dist_y.pdf(v_centroid_y)

        p = f_X_U * f_Y_U
        q = f_X_TN * f_Y_TN
        v_weight = p / q

        v_weight_norm = v_weight/v_weight.sum()
    else:
        v_weight = 1
        v_weight_norm = 1/num_simulations

    # Dataframe of centroids, depths, and weights
    df_storm_sample = pd.DataFrame(dict(
        x_sampled = v_centroid_x,
        y_sampled = v_centroid_y,
        prob = v_weight_norm,
        weight = v_weight,
    ))

    return df_storm_sample


def sample_storm_centers_lhs(v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    xmin, ymin = v_domain_stats.minx, v_domain_stats.miny
    xrange, yrange = v_domain_stats.range_x, v_domain_stats.range_y

    if dist_x is None:
        # Uniform sampling
        dist_x = uniform(xmin, xrange)
        dist_y = uniform(ymin, yrange)

        v_centroid_x = dist_x.rvs(num_simulations)
        v_centroid_y = dist_y.rvs(num_simulations)
        v_weight = 1
        v_weight_norm = 1 / num_simulations

    else:
        # LHS for truncated normal
        unit_samples = lhs(2, samples=num_simulations)

        def lhs_truncnorm(unit_vals, dist):
            values = dist.ppf(unit_vals)
            pdf_vals = dist.pdf(values)
            return values, pdf_vals

        v_centroid_x, f_x_vals = lhs_truncnorm(unit_samples[:, 0], dist_x)
        v_centroid_y, f_y_vals = lhs_truncnorm(unit_samples[:, 1], dist_y)

        # Importance sampling weights
        f_X_U = 1 / xrange
        f_Y_U = 1 / yrange
        p = f_X_U * f_Y_U
        q = f_x_vals * f_y_vals
        v_weight = np.where(q > 0, p / q, 0)
        v_weight_norm = v_weight / np.sum(v_weight)

    return pd.DataFrame({
        "x_sampled": v_centroid_x,
        "y_sampled": v_centroid_y,
        "prob": v_weight_norm,
        "weight": v_weight,
    })



#Get storm catalogue samples
def sample_storm_catalogues(df_storms: pd.DataFrame, num_simulations=10000) -> pd.DataFrame:
    return df_storms.sample(num_simulations, replace=True).reset_index(drop=True)


def sample_storms(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)
    
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers(v_domain_stats=v_domain_stats, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)
    
    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample


def sample_storms_lhs(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)
    
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers_lhs(v_domain_stats=v_domain_stats, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)
    
    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample


def sample_storms_lhs_equally(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    unique_storms = df_storms.reset_index(drop=True)
    n_storms = len(unique_storms)

    if num_simulations < n_storms:
        raise ValueError(f"num_simulations ({num_simulations}) must be ≥ number of unique storms ({n_storms})")

    reps = num_simulations // n_storms
    remainder = num_simulations % n_storms

    repeated = pd.concat([unique_storms] * reps, ignore_index=True)

    if remainder > 0:
        extra = unique_storms.sample(remainder, replace=False)
        repeated = pd.concat([repeated, extra], ignore_index=True)

    _df_storm_sample = repeated
    
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers_lhs(v_domain_stats=v_domain_stats, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)
    
    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample


def sample_uniform_centers(v_domain_stats: pd.Series, num_simulations: int) -> pd.DataFrame:
    dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
    dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)

    v_centroid_x = dist_x.rvs(num_simulations)
    v_centroid_y = dist_y.rvs(num_simulations)

    return pd.DataFrame({
        'x_sampled': v_centroid_x,
        'y_sampled': v_centroid_y,
        'weight': 1.0,
        'prob': 1.0 / num_simulations
    })

def sample_truncated_normal_centers(v_domain_stats: pd.Series, dist_x, dist_y, num_simulations: int) -> pd.DataFrame:

    # Sample from proposal (truncated normal)
    x_sampled = dist_x.rvs(num_simulations)
    y_sampled = dist_y.rvs(num_simulations)

    # Evaluate PDF under target (uniform) and proposal (truncnorm)
    f_x_uniform = 1 / v_domain_stats.range_x
    f_y_uniform = 1 / v_domain_stats.range_y

    f_x_trunc = dist_x.pdf(x_sampled)
    f_y_trunc = dist_y.pdf(y_sampled)

    # Compute importance weights: p(x)/q(x)
    weights = (f_x_uniform * f_y_uniform) / (f_x_trunc * f_y_trunc)
    weights = np.where((f_x_trunc > 0) & (f_y_trunc > 0), weights, 0.0)

    # Normalize weights to get probabilities
    weights_norm = weights / np.sum(weights)

    return pd.DataFrame({
        'x_sampled': x_sampled,
        'y_sampled': y_sampled,
        'weight': weights,
        'prob': weights_norm
    })



def sample_storm_centers_bivariate_rejection(v_domain_stats: pd.Series,
                                             mu: np.ndarray,
                                             sigma: np.ndarray,
                                             num_simulations: int = 10000,
                                             max_iter: int = 1_000_000) -> pd.DataFrame:
    """
    Rejection sampling from a truncated bivariate normal.

    Args:
        v_domain_stats (pd.Series): Domain bounds with minx, miny, range_x, range_y.
        mu (np.ndarray): Mean (2,)
        sigma (np.ndarray): Covariance (2x2)
        num_simulations (int): Number of samples desired.
        max_iter (int): Max number of attempts before failing.

    Returns:
        pd.DataFrame: DataFrame with x_sampled, y_sampled, prob, weight.
    """
    minx, miny = v_domain_stats.minx, v_domain_stats.miny
    maxx = minx + v_domain_stats.range_x
    maxy = miny + v_domain_stats.range_y

    mvn = multivariate_normal(mean=mu, cov=sigma)
    accepted = []

    n_accepted = 0
    attempts = 0

    while n_accepted < num_simulations and attempts < max_iter:
        sample = mvn.rvs(size=1000)
        in_bounds = sample[
            (sample[:, 0] >= minx) & (sample[:, 0] <= maxx) &
            (sample[:, 1] >= miny) & (sample[:, 1] <= maxy)
        ]

        accepted.append(in_bounds)
        n_accepted += len(in_bounds)
        attempts += 1000

    if n_accepted < num_simulations:
        raise RuntimeError(f"Only {n_accepted} samples accepted after {attempts} attempts.")

    accepted = np.vstack(accepted)[:num_simulations]

    # Importance weights
    f_U = 1 / (v_domain_stats.range_x * v_domain_stats.range_y)
    f_T = mvn.pdf(accepted)
    weights = f_U / f_T
    weights[f_T == 0] = 0
    weights_norm = weights / weights.sum()

    return pd.DataFrame({
        'x_sampled': accepted[:, 0],
        'y_sampled': accepted[:, 1],
        'prob': weights_norm,
        'weight': weights
    })

def sample_storms_bivariate(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)
    
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers_bivariate_rejection(v_domain_stats=v_domain_stats, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)
    
    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample