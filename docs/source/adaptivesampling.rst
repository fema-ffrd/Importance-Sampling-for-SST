Adaptive Importance Sampling
============================

This module provides **Adaptive Importance Sampling (AIS)** for Stochastic Storm Transposition (SST).  
AIS iteratively updates sampling distributions, allowing the sampler to concentrate on the most informative storm placements.  

The adaptive sampler uses a **mixture of truncated Gaussians**, with parameters updated across iterations 
based on precipitation response and a user-defined reward function.

---

Usage
-----

.. code-block:: python

   from SSTImportanceSampling import (
       Preprocessor,
       ImportanceSampler,
       StormDepthProcessor,
       AdaptParams,
       AdaptiveMixtureSampler
   )

   trinity = Preprocessor.load(
       config_path="/workspaces/Importance-Sampling-for-SST/data/1_interim/Trinity/config.json"
   )

---

Defining Initial Parameters
---------------------------

Adaptive sampling begins with an initial set of distribution parameters, 
provided via :class:`AdaptParams`.  

.. code-block:: python

   import numpy as np
   from SSTImportanceSampling import AdaptParams

   params = AdaptParams(
       mu_x_n = trinity.watershed_stats["x"],
       mu_y_n = trinity.watershed_stats["y"],
       sd_x_n = trinity.watershed_stats["range_x"],
       sd_y_n = trinity.watershed_stats["range_y"],

       mu_x_w = trinity.domain_stats["x"],
       mu_y_w = trinity.domain_stats["y"],
       sd_x_w = trinity.domain_stats["range_x"] / np.sqrt(12),
       sd_y_w = trinity.domain_stats["range_y"] / np.sqrt(12),

       rho_n = 0,     # narrow correlation
       rho_w = 0,     # wide correlation
       mix   = 0.8,   # initial mixture weight for narrow
       alpha = 0.75,  # adaptation step size
   )

---

Running the Adaptive Sampler
----------------------------

The adaptive sampler refines distribution parameters over several iterations, 
guided by precipitation response and reward thresholds.  

.. code-block:: python

   sampler = AdaptiveMixtureSampler(
       data=trinity,
       params=params,
       precip_cube=trinity.cumulative_precip,
       seed=42
   )

   # Adapt parameters over 10 iterations
   history = sampler.adapt(
       num_iterations=10,
       samples_per_iter=1000,
       depth_threshold=50.8
   )

---

Final Sampling
--------------

After adaptation, a final set of samples can be drawn using the tuned mixture distribution.  

.. code-block:: python

   final_samples = sampler.sample_final(
       n=13_000,
       num_realizations=50,
       with_depths=True
   )

---

Outputs
-------

- **History** : pandas.DataFrame of parameter evolution during adaptation.
- **Final samples** : GeoDataFrame of storm centers with associated weights and (optionally) depths.  