from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import uniform
import geopandas as gpd
from scipy.stats import multivariate_normal, uniform

def compute_rho_from_storms(df_storms: pd.DataFrame) -> float:
    """
    Compute the Pearson correlation coefficient (rho) between x and y storm centers.

    Args:
        df_storms (pd.DataFrame): DataFrame with 'x' and 'y' columns for storm centroids.

    Returns:
        float: Correlation coefficient rho between x and y.
    """
    if not {'x', 'y'}.issubset(df_storms.columns):
        raise ValueError("DataFrame must contain 'x' and 'y' columns.")

    rho = np.corrcoef(df_storms['x'], df_storms['y'])[0, 1]
    return rho

def sample_storm_centers(v_domain_stats: pd.Series,
                         v_watershed_stats: pd.Series = None,
                         method: str = "Uniform",
                         sigma_scale: float = 1.0,
                         rho: float = 0.0,
                         num_simulations: int = 10000,
                         max_iter: int = 1_000_000) -> pd.DataFrame:
    """
    Sample storm centers using uniform or truncated bivariate normal distribution.

    Args:
        v_domain_stats (pd.Series): Domain bounds with minx, miny, range_x, range_y.
        v_watershed_stats (pd.Series): Watershed stats with x, y, range_x, range_y (required for BVN).
        method (str): "Uniform" or "BVN".
        sigma_scale (float): Scale factor for std devs in BVN.
        rho (float): Correlation coefficient between x and y.
        num_simulations (int): Number of samples to generate.
        max_iter (int): Max iterations for rejection sampling in BVN.

    Returns:
        pd.DataFrame: DataFrame with x_sampled, y_sampled, weight, prob.
    """
    minx, miny = v_domain_stats.minx, v_domain_stats.miny
    maxx = minx + v_domain_stats.range_x
    maxy = miny + v_domain_stats.range_y

    if method == "Uniform":
        dist_x = uniform(minx, v_domain_stats.range_x)
        dist_y = uniform(miny, v_domain_stats.range_y)

        x_sampled = dist_x.rvs(num_simulations)
        y_sampled = dist_y.rvs(num_simulations)

        weight = 1
        prob = 1 / num_simulations

        return pd.DataFrame({
            "x_sampled": x_sampled,
            "y_sampled": y_sampled,
            "weight": weight,
            "prob": prob
        })

    elif method == "BVN":
        if v_watershed_stats is None:
            raise ValueError("v_watershed_stats must be provided when method='BVN'")

        mu = np.array([v_watershed_stats.x, v_watershed_stats.y])
        sx = v_watershed_stats.range_x * sigma_scale
        sy = v_watershed_stats.range_y * sigma_scale

        sigma = np.array([
            [sx ** 2, rho * sx * sy],
            [rho * sx * sy, sy ** 2]
        ])

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

        f_U = 1 / (v_domain_stats.range_x * v_domain_stats.range_y)
        f_T = mvn.pdf(accepted)
        weights = f_U / f_T
        weights[f_T == 0] = 0
        weights_norm = weights / weights.sum()

        return pd.DataFrame({
            "x_sampled": accepted[:, 0],
            "y_sampled": accepted[:, 1],
            "prob": weights_norm,
            "weight": weights
        })

    else:
        raise ValueError("method must be either 'Uniform' or 'BVN'")

#Get storm catalogue samples
def sample_storm_catalogues(df_storms: pd.DataFrame, num_simulations=10000) -> pd.DataFrame:
    return df_storms.sample(num_simulations, replace=True).reset_index(drop=True)


def sample_storms(df_storms: pd.DataFrame,
                  v_domain_stats: pd.Series,
                  v_watershed_stats: pd.Series = None,
                  method: str = "Uniform",
                  sigma_scale: float = 1.0,
                  rho: float = 0.0,
                  num_simulations: int = 10000) -> pd.DataFrame:
    """
    Sample storms and apply spatial transposition.

    Args:
        df_storms (pd.DataFrame): Catalog of storms with 'x', 'y' columns.
        v_domain_stats (pd.Series): Bounding box of the transposition domain.
        v_watershed_stats (pd.Series): Watershed stats (required if method='BVN').
        method (str): "Uniform" or "BVN".
        sigma_scale (float): Std dev scale for BVN.
        rho (float): Correlation coefficient for BVN.
        num_simulations (int): Number of simulations.

    Returns:
        pd.DataFrame: Storm catalog with new centroids, deltas, weights, and probabilities.
    """
    # Sample storm properties from catalog
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)

    # Sample storm centers (uniform or bivariate)
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers(
        v_domain_stats=v_domain_stats,
        v_watershed_stats=v_watershed_stats,
        method=method,
        sigma_scale=sigma_scale,
        rho=rho,
        num_simulations=num_simulations
    )

    # Combine sampled storms with sampled centers and compute offsets
    df_storm_sample = (
        pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del=lambda df: df.x_sampled - df.x)
        .assign(y_del=lambda df: df.y_sampled - df.y)
    )

    return df_storm_sample


