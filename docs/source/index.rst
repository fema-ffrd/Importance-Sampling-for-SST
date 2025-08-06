SST Importance Sampling Documentation
=====================================

This documentation provides an overview of the modules used for sampling storm transposition centers
and preprocessing precipitation data using the SST (Stochastic Storm Transposition) framework.

It includes:

- A preprocessing module for reading storm catalog data, computing storm centers, and saving structured output
- A flexible sampling module for generating storm centers using uniform and Gaussian copula-based importance sampling
- A storm depth processor for calculating average precipitation and return periods for transposed storms

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   usage

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api_preprocessor
   api_importancesampler
   api_stormdepthprocessor