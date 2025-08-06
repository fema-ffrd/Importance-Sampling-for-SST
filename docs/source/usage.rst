User Guide
==========

This section walks through how to use the `SSTImportanceSampling` library for preprocessing, sampling storm centers, and computing precipitation statistics with transposed storms.

Quick Start
-----------

Install the library in editable mode (recommended for development):

.. code-block:: bash

   git clone https://github.com/fema-ffrd/Importance-Sampling-for-SST.git
   cd Importance-Sampling-for-SST
   pip install -e .

Once installed, you can import the main components:

.. code-block:: python

   import SSTImportanceSampling

Make sure your environment has the required dependencies (these are installed automatically if using `pip`):

- numpy
- pandas
- geopandas
- rasterio
- xarray
- shapely
- joblib
- tqdm

Preprocessing
-------------

.. include:: preprocessor_usage.rst

Sampling Storm Centers
----------------------

.. include:: sampler_usage.rst

Calculate Storm Depths
-----------------------

.. include:: stormdepthprocessor_usage.rst
