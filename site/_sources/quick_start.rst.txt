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
Example input data structure is provided in the ``example-input-data`` folder located at the root of the repository. The watershed and domain geojsons and the dss catalog will need to be replaced by your data and the config file should reflect the same.  


Dependencies
------------

The following Python libraries are required and will be installed automatically:

- pip
- geopandas
- joblib
- jupyter
- matplotlib
- numpy
- pandas
- plotly
- plotnine
- pyproj
- rasterio
- rasterstats
- rioxarray
- scikit-learn
- scipy
- seaborn
- shapely
- statsmodels
- tqdm
- typing-extensions
- xarray
- zarr
- dask
- s3fs
- hecdss
