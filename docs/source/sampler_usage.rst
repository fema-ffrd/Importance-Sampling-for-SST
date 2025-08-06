Use the Sampler
~~~~~~~~~~~~~~~

You can now generate storm center samples using the :class:`SSTImportanceSampling.importancesampler.ImportanceSampler` class.
Choose a distribution and provide the required parameters.

Example 1: **Gaussian Copula Sampling**

.. code-block:: python

    from SSTImportanceSampling.sampler import Sampler

    sampler = Sampler(
        distribution="gaussian_copula",
        params={"scale_sd_x": 0.25, "scale_sd_y": 0.25, "rho": 0.5},
        num_simulations=500,
        num_rep=5,
    )
    samples_df = sampler.sample(pre.domain_gdf, pre.watershed_gdf)

Example 2: **Uniform Sampling**

For uniform sampling, `params` can be omitted or passed as an empty dictionary:

.. code-block:: python

    sampler = Sampler(
        distribution="uniform",
        params={},
        num_simulations=500,
        num_rep=5,
    )
    samples_df = sampler.sample(pre.domain_gdf, pre.watershed_gdf)

The returned `samples_df` will contain:
- Storm center coordinates (`x`, `y`)
- Repetition index (`rep`)
- Event ID (`event_id`)
- Sampling weights (`weight`)
