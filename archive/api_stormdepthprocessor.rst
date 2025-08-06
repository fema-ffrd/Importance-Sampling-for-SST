StormDepthProcessor
====================

Module for computing watershed-averaged precipitation depths and corresponding return periods
from a catalog of transposed storm samples.

Use :class:`SSTImportanceSampling.stormdepthprocessor.StormDepthProcessor` to evaluate the hydrologic impact
of shifted storms over a watershed polygon.

The class supports:
- Shifting storm footprints based on sampled storm center coordinates
- Masking shifted precipitation arrays by the watershed geometry
- Computing average watershed precipitation for each storm
- Weighting storms and calculating return periods using arrival rate theory
- Parallelized processing using `joblib`

After creating a `StormDepthProcessor` instance, call :meth:`SSTImportanceSampling.stormdepthprocessor.StormDepthProcessor.shift_and_extract_precip`
with a DataFrame of sampled storm centers to get watershed-averaged precipitation and return periods.

Typical workflow:
1. Initialize the processor with the storm precipitation cube, storm origin centers, and watershed geometry.
2. Call `shift_and_extract_precip()` with storm samples.
3. (Optional) Use `add_return_periods()` to add exceedance probabilities and return periods to custom storm lists.

.. automodule:: SSTImportanceSampling.stormdepthprocessor
   :members:
   :undoc-members:
   :show-inheritance: