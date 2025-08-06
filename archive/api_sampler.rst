Sampler
=======

Module for generating storm center samples using importance sampling or uniform sampling.

Use :class:`SSTImportanceSampling.sampler.Sampler` to create storm center samples.

The class supports:
- Uniform sampling within a transposition region
- Gaussian copula-based importance sampling centered on the watershed
- User-defined number of simulations and repetitions
- Dynamic parameter validation based on selected distribution

After creating a `Sampler` instance, call :meth:`SSTImportanceSampling.sampler.Sampler.sample` with the domain and watershed geometries.

.. automodule:: SSTImportanceSampling.sampler
    :members:
    :undoc-members:
    :show-inheritance: