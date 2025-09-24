Preprocessing
=============

The preprocessing module handles reading storm catalog DSS files, extracting storm centers, and saving preprocessed datasets (cumulative precipitation, storm centers, reprojected basin and domain).

Steps
-----

1. Read a config.json file that contains the locations of input data.
2. Reproject the basin and domain shapefiles to the SHG projection.
3. Compute storm centers based on cumulative precipitation or time-windowed grids.
4. Save intermediate files to a NetCDF and update the configuration file.

Example
-------

.. code-block:: python

   from SSTImportanceSampling import Preprocessor

   trinity = Preprocessor(config_path = "data/0_source/Trinity/config.json", output_folder = "data/1_interim/Trinity")
   trinity.run()

Output Files
--------------

- Reprojected GeoPackages (watershed and domain)
- NetCDF file with precipitation sum (over time) for each storm, stored as a 3D array (storm_path, x, y).
- parquet file with storm centers

Configuration Notes
-------------------

The input `config.json` file should contain paths to:

- Folder location where storm DSS files are stored 
- Watershed geojson location
- Transposition Domain geojson location