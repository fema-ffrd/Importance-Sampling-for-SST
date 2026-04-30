"""Geometry processing for SST Importance Sampling."""

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd
from pyproj import CRS

SHG_WKT = (
    'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",'
    'GEOGCS["GCS_North_American_1983",'
    'DATUM["D_North_American_1983",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["Central_Meridian",-96.0],'
    'PARAMETER["Standard_Parallel_1",29.5],'
    'PARAMETER["Standard_Parallel_2",45.5],'
    'PARAMETER["Latitude_Of_Origin",23.0],'
    'UNIT["Meter",1.0]]'
)

SHG_CRS = CRS.from_wkt(SHG_WKT)


def load_and_project_to_shg(
    watershed_path: str | Path,
    domain_path: str | Path,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load and project watershed and domain geometries to SHG CRS.

    Parameters
    ----------
    watershed_path : str | Path
        Path to watershed GeoJSON file.
    domain_path : str | Path
        Path to domain/transposition region GeoJSON file.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        Projected watershed and domain GeoDataFrames in SHG CRS.

    Raises
    ------
    FileNotFoundError
        If input files do not exist.
    ValueError
        If geometries do not have CRS defined.

    Examples
    --------
    >>> watershed, domain = load_and_project_to_shg(
    ...     "watershed.geojson", "domain.geojson"
    ... )
    """
    watershed_path = Path(watershed_path)
    domain_path = Path(domain_path)

    if not watershed_path.exists():
        raise FileNotFoundError(f"Watershed file not found: {watershed_path}")

    if not domain_path.exists():
        raise FileNotFoundError(f"Domain file not found: {domain_path}")

    watershed = gpd.read_file(watershed_path)
    domain = gpd.read_file(domain_path)

    if watershed.crs is None:
        raise ValueError("Watershed GeoJSON must have CRS defined")

    if domain.crs is None:
        raise ValueError("Domain GeoJSON must have CRS defined")

    watershed = watershed.to_crs(SHG_CRS)
    domain = domain.to_crs(SHG_CRS)

    return watershed, domain


def compute_spatial_stats(gdf: gpd.GeoDataFrame, name: str = "geometry") -> pd.Series:
    """
    Compute comprehensive spatial statistics for a geometry.

    Computes bounding box bounds, centroid, spatial extents, and area.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry column.
    name : str, optional
        Name identifier for this geometry, by default "geometry".

    Returns
    -------
    pd.Series
        Series containing spatial statistics with keys:
        - name: Geometry identifier
        - minx, miny: Southwest corner of bounding box
        - maxx, maxy: Northeast corner of bounding box
        - centroid_x, centroid_y: Centroid coordinates
        - range_x: East-West extent (maxx - minx)
        - range_y: North-South extent (maxy - miny)
        - area: Total area in square units

    Examples
    --------
    >>> stats = compute_spatial_stats(watershed_gdf, "watershed")
    >>> print(stats["centroid_x"], stats["centroid_y"])
    """
    geom_union = gdf.geometry.unary_union
    minx, miny, maxx, maxy = gdf.total_bounds
    centroid = geom_union.centroid
    area = geom_union.area

    stats = pd.Series(
        {
            "name": name,
            "minx": float(minx),
            "miny": float(miny),
            "maxx": float(maxx),
            "maxy": float(maxy),
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "range_x": float(maxx - minx),
            "range_y": float(maxy - miny),
            "area": float(area),
        }
    )

    return stats