# geometry.py

import geopandas as gpd
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


def load_and_project_to_shg(watershed_path, domain_path):
    shg = CRS.from_wkt(SHG_WKT)

    watershed = gpd.read_file(watershed_path)
    domain = gpd.read_file(domain_path)

    if watershed.crs is None or domain.crs is None:
        raise ValueError("GeoJSON must have CRS defined.")

    watershed = watershed.to_crs(shg)
    domain = domain.to_crs(shg)

    return watershed, domain


def compute_centroids(watershed, domain):
    w_geom = watershed.geometry.unary_union
    d_geom = domain.geometry.unary_union

    return w_geom.centroid, d_geom.centroid