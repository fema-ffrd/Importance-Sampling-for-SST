Preprocessing
=============

The preprocessing module handles reading storm catalog files, extracting storm centers, and saving preprocessed datasets.

Steps
-----

1. Read a STAC catalog or storm folder.
2. Reproject the basin and domain shapefiles to the SHG projection.
3. Compute storm centers based on cumulative precipitation or time-windowed grids.
4. Save intermediate files to a NetCDF and update the configuration.

Example
-------

.. code-block:: python

   from SSTImportanceSampling import Preprocessor

   pp = Preprocessor(config_path="config.json")
   pp.run_all()

Typical Output
--------------

- Reprojected GeoPackages
- NetCDF file with cumulative precipitation
- GeoJSON with storm centers

Configuration Notes
-------------------

The `config.json` file should contain paths to:

- STAC catalog or local DSS folder
- Basin and transposition domain shapefiles
- Output filenames for NetCDF and storm centers
