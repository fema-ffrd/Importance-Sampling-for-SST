Importance Sampling
===================

This module provides storm center sampling for Stochastic Storm Transposition (SST) using several proposal distributions.  

Supported methods include:

- **Uniform sampling**
- **Truncated Gaussian**
- **Gaussian Copula**
- **Mixture of Truncated Gaussians**

Each method produces sampled storm centers which can then be evaluated with the
:class:`StormDepthProcessor`.

Usage
-----

.. code-block:: python

   from SSTImportanceSampling import Preprocessor, ImportanceSampler, StormDepthProcessor

   trinity = Preprocessor.load(
       config_path="/workspaces/Importance-Sampling-for-SST/data/1_interim/Trinity/config.json"
   )

Examples
--------

Uniform Sampling
~~~~~~~~~~~~~~~~
Uniform sampling places storm centers evenly across all valid locations.  

.. code-block:: python

   sampler = ImportanceSampler(
       distribution="uniform",
       params={},
       num_simulations=20_000,
       num_realizations=50
   )

   uniform_samples = sampler.sample(data=trinity)
   uniform_depths = StormDepthProcessor(trinity).run(uniform_samples, n_jobs=-1)


Truncated Gaussian Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Centers are sampled around the watershed centroid with user-defined spread.  

.. code-block:: python

   params = {
       "mu_x": trinity.watershed_stats["x"],
       "mu_y": trinity.watershed_stats["y"],
       "sd_x": trinity.watershed_stats["range_x"] * 0.7,
       "sd_y": trinity.watershed_stats["range_y"] * 0.35,
   }

   sampler = ImportanceSampler(
       distribution="truncated_gaussian",
       params=params,
       num_simulations=15_000,
       num_realizations=50,
   )

   tn_samples = sampler.sample(data=trinity)
   tn_depths = StormDepthProcessor(trinity).run(tn_samples, n_jobs=-1)


Gaussian Copula Sampling
~~~~~~~~~~~~~~~~~~~~~~~~
Adds correlation between x and y dimensions using a Gaussian copula.  

.. code-block:: python

   params = {
       "mu_x": trinity.watershed_stats["x"],
       "mu_y": trinity.watershed_stats["y"],
       "sd_x": trinity.watershed_stats["range_x"] * 0.6,
       "sd_y": trinity.watershed_stats["range_y"] * 0.35,
       "rho": -0.1,
   }

   sampler = ImportanceSampler(
       distribution="gaussian_copula",
       params=params,
       num_simulations=15_000,
       num_realizations=50,
   )

   copula_samples = sampler.sample(data=trinity)
   copula_depths = StormDepthProcessor(trinity).run(copula_samples, n_jobs=-1)


Mixture of Truncated Gaussians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Combines a narrow distribution around the watershed with a wide distribution over the full domain.  

.. code-block:: python

   params = {
       "mu_x_narrow": trinity.watershed_stats["x"],
       "mu_y_narrow": trinity.watershed_stats["y"],
       "mu_x_wide": trinity.domain_stats["x"],
       "mu_y_wide": trinity.domain_stats["y"],
       "sd_x_narrow": trinity.watershed_stats["range_x"] * 0.05,
       "sd_y_narrow": trinity.watershed_stats["range_y"] * 0.05,
       "sd_x_wide": trinity.domain_stats["range_x"] / np.sqrt(12),
       "sd_y_wide": trinity.domain_stats["range_y"] / np.sqrt(12),
       "mix": 0.95,
       "rho_narrow": -0.5,
       "rho_wide": 0,
   }

   sampler = ImportanceSampler(
       distribution="mixture_trunc_gauss",
       params=params,
       num_simulations=6_000,
       num_realizations=50,
   )

   mixture_samples = sampler.sample(data=trinity)
   mixture_depths = StormDepthProcessor(trinity).run(mixture_samples, n_jobs=-1)


Outputs
-------
Each method returns a **GeoDataFrame** of sampled storm centers with associated weights.  
Results can be evaluated using :class:`StormDepthProcessor` to compute watershed-average depths
and exceedance probabilities.
