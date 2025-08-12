Quick Start
===========

Install the library in editable mode (recommended for development):

.. code-block:: bash

   git clone https://github.com/fema-ffrd/Importance-Sampling-for-SST.git
   cd Importance-Sampling-for-SST
   pip install -e .

Once installed, you can import the library:

.. code-block:: python

   import SSTImportanceSampling


Example Data
------------
Example datasets are provided in the ``example-input-data`` folder located at the root of the rerepository. These can be used to test and explore the functionality of the library. However, the catalog doesn't include a full set of storm events. 


Dependencies
------------

The following Python libraries are required and will be installed automatically:

- numpy
- pandas
- geopandas
- rasterio
- xarray
- shapely
- joblib
- tqdm
- scipy
- typing
- hecdss
- pyproj
- affine
