This folder contains a sample storm catalog dataset covering a 2-year period. 

In actual applications, DSS files should represent storms across the entire period of record.

------------------------------------------
Contents Required for Importance Sampling
------------------------------------------
To run importance sampling, ensure the following files are available and correctly referenced in config.json:

Watershed GeoJSON – Defines the watershed boundary.
Transposition Domain GeoJSON – Specifies the domain for storm transposition.
Storm Catalog Folder – Contains individual DSS files for each storm.

-----
Notes
-----
This dataset is for demonstration purposes only.
Update config.json to reflect the correct file paths before use.