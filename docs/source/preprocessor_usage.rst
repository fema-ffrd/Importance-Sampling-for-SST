Prepare your config.json
~~~~~~~~~~~~~~~~~~~~~~~~

Ensure your JSON file includes paths to:
- Watershed GeoJSON
- Transposition region GeoJSON
- DSS catalog folder

Example:
::

    {
        "watershed": {
            "geometry_file": "path/to/watershed.geojson"
        },
        "transposition_region": {
            "geometry_file": "path/to/domain.geojson"
        },
        "catalog": {
            "catalog_folder": "path/to/dss/folder"
        }
    }

Run the Preprocessor
~~~~~~~~~~~~~~~~~~~~

Use the following in your script or notebook:

.. code-block:: python

    from SSTImportanceSampling.preprocessor import Preprocessor

    pre = Preprocessor("path/to/config.json", "path/to/output_folder")
    pre.run()

This will:
- Reproject the watershed and domain
- Compute cumulative precipitation from all DSS files
- Compute storm centers
- Save everything in the output folder

Load a Preprocessed Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you've already run preprocessing, load results with:

.. code-block:: python

    pre = Preprocessor.load("path/to/output_folder")

This skips recomputation and loads:
- Watershed and domain (GPKG)
- Cumulative precipitation (NetCDF)
- Storm centers (Parquet)
