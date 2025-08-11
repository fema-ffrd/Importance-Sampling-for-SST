Calculate Storm Depths
=======================

This module computes average precipitation depth over a target basin and derives return periods.

Features
--------

- Calculate basin-average precipitation for each transposed storm
- Estimate return periods using exceedance probabilities
- Use Langbein’s formula assuming a Poisson arrival rate

Example
-------

.. code-block:: python

   from SSTImportanceSampling import StormDepthProcessor

   dp = StormDepthProcessor(config_path="config.json")
   df = dp.compute_all()

Langbein Return Period Formula
------------------------------

Return period is estimated as:

.. math::

   RP = \frac{1 + N}{\text{rank}}

Where `N` is the number of storms and `rank` is assigned from largest to smallest depth.

Output
------

- DataFrame with storm ID, depth, and return period
- CSV or plot of depth-frequency curves

Tips
----

- You can specify custom basin polygons
- Use smoothing or bootstrapping for uncertainty in RP estimates
