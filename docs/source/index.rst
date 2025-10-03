SST Importance Sampling Documentation
=====================================

This user guide provides a high level overview for using Importance Sampling of x and y coordinates for Stochastic Storm Transposition aimed at reducing the number of simulations required to achieve rare event scenarios. 

It includes:

- A preprocessing module for reading storm catalog data, computing storm centers, valid tranpositions and saving structured output
- Flexible sampling modules for generating storm centers and importance weights using importance sampling (including vanilla and adaptive sampling strategies)
- A storm depth processor for evaluating average precipitation and exceedance prbabilities for transposed storms

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   quick_start
   preprocessing
   importancesampling
   adaptivesampling

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api_preprocessor
   api_importancesampler
   api_adaptiveimportancesampler
   api_stormdepthprocessor