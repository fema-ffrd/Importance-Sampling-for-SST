This guide provides a step-by-step walkthrough for using the
:class:`SSTImportanceSampling.stormdepthprocessor.StormDepthProcessor` class
to compute watershed-averaged precipitation depths and estimate return periods
for sampled storms.

Overview
~~~~~~~~

The `StormDepthProcessor` takes as input:
- A 3D precipitation cube (`xarray.DataArray`) with dimensions `(storm_path, y, x)`
- A DataFrame of original storm center locations with `storm_path`, `x`, `y`
- A watershed polygon (`GeoDataFrame`) in the same CRS as the precipitation cube

It computes:
- Transposed storm footprints using sampled center shifts
- Average precipitation over the watershed
- Return periods using storm weights and a Poisson arrival model

Typical Workflow
~~~~~~~~~~~~~~~~

1. **Load preprocessed data**

Use the :class:`SSTImportanceSampling.preprocessor.Preprocessor` to prepare input files.

.. code-block:: python

    from SSTImportanceSampling.preprocessor import Preprocessor

    pre = Preprocessor.load("outputs/")
    precip_cube = pre.cumulative_precip
    storm_centers = pre.storm_centers
    watershed_gdf = pre.watershed_gdf

2. **Initialize the processor**

.. code-block:: python

    from SSTImportanceSampling.stormdepthprocessor import StormDepthProcessor

    processor = StormDepthProcessor(
        precip_cube=precip_cube,
        storm_centers=storm_centers,
        watershed_gdf=watershed_gdf,
        arrival_rate=10  # storms/year
    )

3. **Run precipitation extraction on storm samples**

Use a sampled storm DataFrame (e.g., from :class:`Sampler`) with `x`, `y`, `weight`, `event_id`, `rep` and `storm_path`.

.. code-block:: python

    result_df = processor.shift_and_extract_precip(
        df_storms=samples_df,  # includes storm_path, x, y, weight, rep, etc.
        n_jobs=-1,             # use all cores
        seed=42                # reproducibility
    )

4. **Inspect the result**

Each row in the output contains:
- Event ID
- Shifted storm center coordinates
- Precipitation depth over watershed
- Weight and return period

.. code-block:: python

    result_df.head()

    # Columns: event_id, x, y, weight, storm_path, precip_avg_mm, rep, return_period

5. **Add return periods separately (optional)**

If you want to recompute return periods after filtering/modifying results:

.. code-block:: python

    result_with_rps = processor.add_return_periods(result_df)

Notes
~~~~~

- Input geometries must be in the same projected CRS (e.g., SHG).
- The `precip_cube` must have dimensions named exactly as: `storm_path`, `y`, `x`.
- `samples_df` must include a `storm_path` column that matches one of the original DSS filenames.
- Storm transposition is performed by integer-shifting the storm grid based on dx/dy between centers.

See Also
~~~~~~~~

- :class:`SSTImportanceSampling.preprocessor.Preprocessor`
- :class:`SSTImportanceSampling.sampler.Sampler`
