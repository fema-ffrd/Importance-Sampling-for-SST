SST Importance Sampling Documentation
=====================================

This user guide provides a high level overview for using Importance Sampling of x and y coordinates for Stochastic Storm Transposition aimed at reducing the number of simulations required to achieve rare event scenarios. 

It includes:

- A preprocessing module for reading storm catalog data, computing storm centers, and saving structured output
- Flexible sampling modules for generating storm centers using importance sampling (including vanilla and adaptive sampling strategies)
- A storm depth processor for evaluating average precipitation and return periods for transposed storms

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   quick_start
   preprocessing
   sampling
   depths

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api_preprocessor
   api_importancesampler
   api_stormdepthprocessor