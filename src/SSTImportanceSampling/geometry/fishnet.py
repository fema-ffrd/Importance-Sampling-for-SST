import geopandas as gpd
import math
import csv
import os
from pathlib import Path
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep

def create_uniform_fishnet_csv(
    gpkg_path: str,
    spacing: float,
    output_path: str,
    include_boundary: bool = False,
):
    """
    Generate uniform grid points inside polygon and save to CSV or Parquet.

    Parameters
    ----------
    gpkg_path : str
        Path to polygon GeoPackage
    spacing : float
        Grid spacing (same units as CRS)
    output_path : str
        Output file path (can be .csv or .parquet)
    include_boundary : bool
        If True, include boundary points (uses covers instead of contains)
        
    Returns
    -------
    str
        Path to output file
        
    Examples
    --------
    Save as CSV:
        create_uniform_fishnet_csv("domain.gpkg", 1000, "fishnet.csv")
        
    Save as Parquet:
        create_uniform_fishnet_csv("domain.gpkg", 1000, "fishnet.parquet")
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect output format from extension
    output_format = output_path.suffix.lower()
    if output_format not in [".csv", ".parquet"]:
        raise ValueError(
            f"output_path must have .csv or .parquet extension, "
            f"got '{output_format}'"
        )

    gdf = gpd.read_file(gpkg_path)
    polygon = gdf.geometry.iloc[0]

    # Speed optimization for many points
    prepared_polygon = prep(polygon)

    minx, miny, maxx, maxy = polygon.bounds

    xdist = maxx - minx
    ydist = maxy - miny

    xSteps = int(math.floor(abs(xdist) / spacing))
    ySteps = int(math.floor(abs(ydist) / spacing))

    currentYval = maxy + (spacing / 2)

    # Collect points in list
    points = []

    y = 0
    while y < ySteps:

        x = 0
        currentXval = minx + (spacing / 2)

        while x < xSteps:
            x += 1
            currentXval += spacing

            point = Point(currentXval, currentYval)

            if include_boundary:
                inside = prepared_polygon.covers(point)
            else:
                inside = prepared_polygon.contains(point)

            if inside:
                points.append({"x": currentXval, "y": currentYval})

        y += 1
        currentYval -= spacing

    # Convert to DataFrame
    df = pd.DataFrame(points)

    # Save based on format
    if output_format == ".parquet":
        df.to_parquet(output_path, index=False, compression='snappy')
        print(f"✅ Fishnet saved to: {output_path}")
    else:  # .csv
        df.to_csv(output_path, index=False)
        print(f"✅ Fishnet saved to: {output_path}")

    return str(output_path)