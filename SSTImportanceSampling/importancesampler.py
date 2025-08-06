"""
Sampler module to generate storm center samples using various spatial distributions.

Use :class:`Sampler` to generate storm center locations for a given spatial domain.

This module handles:
- Uniform random sampling within a polygon domain
- Gaussian copula-based importance sampling centered on a watershed

Typical usage example::

    sampler = Sampler(
        distribution="gaussian_copula",
        params={"scale_sd_x": 0.2, "scale_sd_y": 0.2, "rho": 0.5},
        num_simulations=1000,
        num_rep=5,
        seed=42,
    )
    df_samples = sampler.sample(domain_gdf, watershed_gdf)

"""

from typing import Literal, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import uniform, truncnorm, norm, multivariate_normal


class ImportanceSampler:
    """
    Sampler class for generating storm center samples using various distributions.

    Supported distributions:
        - 'uniform': Uniform sampling within the domain
        - 'gaussian_copula': Gaussian copula-based importance sampling

    Parameters:
        distribution (Literal["uniform", "gaussian_copula"]): Distribution type
        params (dict): Parameters for the chosen distribution
        num_simulations (int): Number of samples per repetition
        num_rep (int): Number of repetitions
        seed (Optional[int]): Random seed for reproducibility
    """

    def __init__(
        self,
        distribution: Literal["uniform", "gaussian_copula"],
        params: dict,
        num_simulations: int,
        num_rep: int = 1,
        seed: Optional[int] = None,
    ):
        self.distribution = distribution
        self.params = params
        self.num_simulations = num_simulations
        self.num_rep = num_rep
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if self.distribution == "uniform":
            required = set()
        elif self.distribution == "gaussian_copula":
            required = {"scale_sd_x", "scale_sd_y", "rho"}
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        missing = required - self.params.keys()
        if missing:
            raise ValueError(f"Missing required parameters for '{self.distribution}': {missing}")

    def sample(
        self,
        domain_gdf: gpd.GeoDataFrame,
        watershed_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate samples using the selected distribution.

        Args:
            domain_gdf (gpd.GeoDataFrame): Polygon domain for sampling
            watershed_gdf (Optional[gpd.GeoDataFrame]): Required if using 'gaussian_copula'

        Returns:
            pd.DataFrame: Samples with columns ['rep', 'event_id', 'x', 'y', 'weight']
        """
        if self.distribution == "uniform":
            return self._sample_uniform(domain_gdf)
        elif self.distribution == "gaussian_copula":
            if watershed_gdf is None:
                raise ValueError("'watershed_gdf' is required for 'gaussian_copula'")
            return self._sample_gaussian_copula(domain_gdf, watershed_gdf)

    def _sample_uniform(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        bounds = sp_domain.total_bounds
        minx, miny, maxx, maxy = bounds
        range_x = maxx - minx
        range_y = maxy - miny

        dist_x = uniform(minx, range_x)
        dist_y = uniform(miny, range_y)

        all_samples = []
        for rep in range(1, self.num_rep + 1):
            v_x, v_y = [], []
            i = 0
            max_iter = 100

            while len(v_x) < self.num_simulations and i < max_iter:
                _x = dist_x.rvs(self.num_simulations)
                _y = dist_y.rvs(self.num_simulations)
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_x, _y), crs=sp_domain.crs)
                gdf_valid = gdf[gdf.within(sp_domain.geometry.iloc[0])]

                v_x += gdf_valid.geometry.x.tolist()
                v_y += gdf_valid.geometry.y.tolist()
                i += 1

            v_x = v_x[:self.num_simulations]
            v_y = v_y[:self.num_simulations]

            df = pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "x": v_x,
                "y": v_y,
                "weight": np.full(self.num_simulations, 1.0 / (self.num_simulations+1)),
            })
            all_samples.append(df)

        return pd.concat(all_samples, ignore_index=True)

    def _sample_gaussian_copula(
        self, sp_domain: gpd.GeoDataFrame, watershed_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        range_x, range_y = xmax - xmin, ymax - ymin
        mu_x, mu_y = watershed_gdf.geometry.centroid.iloc[0].coords[0]

        sigma_x = self.params["scale_sd_x"] * range_x
        sigma_y = self.params["scale_sd_y"] * range_y
        rho = self.params["rho"]

        a_x, b_x = (xmin - mu_x) / sigma_x, (xmax - mu_x) / sigma_x
        a_y, b_y = (ymin - mu_y) / sigma_y, (ymax - mu_y) / sigma_y

        trunc_x = truncnorm(a_x, b_x, loc=mu_x, scale=sigma_x)
        trunc_y = truncnorm(a_y, b_y, loc=mu_y, scale=sigma_y)

        all_samples = []

        for rep in range(self.num_rep):
            v_x, v_y = [], []
            i = 0
            max_iter = 100

            while len(v_x) < self.num_simulations and i < max_iter:
                u = np.random.uniform(size=(self.num_simulations, 2))
                z = norm.ppf(u)
                L = np.linalg.cholesky([[1, rho], [rho, 1]])
                z_corr = z @ L.T
                u_corr = norm.cdf(z_corr)
                x = trunc_x.ppf(u_corr[:, 0])
                y = trunc_y.ppf(u_corr[:, 1])
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=sp_domain.crs)
                gdf_valid = gdf[gdf.within(sp_domain.geometry.iloc[0])]

                v_x += gdf_valid.geometry.x.tolist()
                v_y += gdf_valid.geometry.y.tolist()
                i += 1

            v_x = v_x[:self.num_simulations]
            v_y = v_y[:self.num_simulations]

            z_x = norm.ppf(trunc_x.cdf(v_x))
            z_y = norm.ppf(trunc_y.cdf(v_y))
            cov = [[1, rho], [rho, 1]]
            phi_bv = multivariate_normal(mean=[0, 0], cov=cov)
            log_q = phi_bv.logpdf(np.stack([z_x, z_y], axis=1))
            log_px = norm.logpdf(z_x)
            log_py = norm.logpdf(z_y)
            log_fx = np.log(trunc_x.pdf(v_x))
            log_fy = np.log(trunc_y.pdf(v_y))

            log_p = -np.log(sp_domain.geometry.iloc[0].area)
            log_weights = log_p - (log_q - log_px - log_py + log_fx + log_fy)
            weights = np.exp(log_weights)
            weights /= weights.sum()

            df = pd.DataFrame({
                "rep": rep + 1,
                "event_id": np.arange(1, self.num_simulations + 1),
                "x": v_x,
                "y": v_y,
                "weight": weights,
            })
            all_samples.append(df)

        return pd.concat(all_samples, ignore_index=True)
