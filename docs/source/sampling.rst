Sampling Storm Centers
=======================

This module provides flexible storm center sampling using:

- Uniform distribution over the domain
- Importance sampling using truncated bivariate normals
- Gaussian copulas
- Mixture models

Usage
-----

.. code-block:: python

   from SSTImportanceSampling import Sampler

   sampler = Sampler(config_path="config.json")
   gdf_samples = sampler.sample(distribution="mixture")

Key Parameters
--------------

- `distribution`: One of `"uniform"`, `"truncated_normal"`, `"copula"`, or `"mixture"`
- `n_samples`: Number of storm centers to draw
- `adaptive`: If `True`, uses adaptive importance sampling (AIS)
- `domain_file`: Polygon GeoJSON or shapefile defining the domain

Outputs
-------

- GeoDataFrame of sampled centers with weights
- CSV or shapefile (optional save)
- Diagnostic plots (optional)

Best Practices
--------------

- Use `adaptive=True` to refine proposals over multiple iterations
- Inspect mixture weights, means, and covariances over time
