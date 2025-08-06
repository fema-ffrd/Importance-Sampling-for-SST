Preprocessor
============

Module to preprocess watershed and storm catalog data for use in SST Importance Sampling.

Use :class:`SSTImportanceSampling.preprocessor.Preprocessor` to preprocess your data.  
Call :meth:`SSTImportanceSampling.preprocessor.Preprocessor.run` after initialization.  
You can reuse a processed dataset using :meth:`SSTImportanceSampling.preprocessor.Preprocessor.load`.

This module handles:
- Reading and projecting watershed/domain geometries
- Reading DSS storm files and computing cumulative precipitation
- Computing optimal storm transposition centers
- Saving preprocessed outputs (NetCDF, GeoPackage, Parquet, updated config)

.. automodule:: SSTImportanceSampling.preprocessor
    :members:
    :undoc-members:
    :show-inheritance: