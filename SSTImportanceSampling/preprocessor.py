from pathlib import Path, PurePosixPath
from typing import List, Dict, Union, Tuple
from urllib.parse import urlparse
import json
import os
import shutil
import tempfile
import requests
import contextlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import xarray as xr
from hecdss import HecDss
from pandas import to_datetime
from pyproj import CRS, Transformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SHG_WKT = 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-96.0],PARAMETER["Standard_Parallel_1",29.5],PARAMETER["Standard_Parallel_2",45.5],PARAMETER["Latitude_Of_Origin",23.0],UNIT["Meter",1.0]]'


class Preprocessor:
    def __init__(
        self,
        catalog_path: Union[str, Path],
        config_path: Union[str, Path],
        shg_wkt: str = SHG_WKT,
    ) -> None:
        self.catalog_path = str(catalog_path)
        self.config_path = str(config_path)
        self.shg_wkt = shg_wkt
        self.storm_centers: List[Dict[str, Union[str, float]]] = []
        self.watershed_gdf: gpd.GeoDataFrame
        self.domain_gdf: gpd.GeoDataFrame
        self.aggregated_precip: xr.DataArray

    def run(self):
        print("[RUN] Getting storm centers...")
        self.storm_centers = self._get_storm_centers()
        print(f"[RUN] Got {len(self.storm_centers)} storm centers")

        print("[RUN] Reading GDF...")
        self.watershed_gdf, self.domain_gdf = self._read_gdf()
        print("[RUN] Got GDFs")

        print("[RUN] Processing all storms to cube...")
        self.aggregated_precip = self._process_all_storms_to_cube()
        print("[RUN] Finished processing storms.")

    def save(self, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save storm centers as Parquet
        pd.DataFrame(self.storm_centers).to_parquet(output_path.with_suffix(".storm_centers.pq"), index=False)

        # Save GeoDataFrames as GeoPackage
        self.watershed_gdf.to_file(output_path.with_suffix(".watershed.gpkg"), driver="GPKG")
        self.domain_gdf.to_file(output_path.with_suffix(".domain.gpkg"), driver="GPKG")

        # Save xarray as NetCDF
        self.aggregated_precip.to_netcdf(output_path.with_suffix(".nc"))

    def _get_storm_centers(self) -> List[Dict[str, Union[str, float]]]:
        storm_info = []
        parsed = urlparse(self.catalog_path)
        is_remote = parsed.scheme in ("http", "https", "s3")
        catalog = pystac.Catalog.from_file(self.catalog_path)

        # Create transformer once (WGS84 -> SHG)
        transformer = Transformer.from_crs("EPSG:4326", CRS.from_wkt(self.shg_wkt), always_xy=True)

        for child_link in catalog.get_links("child"):
            collection = pystac.Collection.from_file(child_link.absolute_href)
            for item in collection.get_all_items():
                props = item.properties
                transform = props.get("aorc:transform", {})
                lat = transform.get("storm_center_lat")
                lon = transform.get("storm_center_lon")
                dss_asset = item.assets.get("dss")
                path = dss_asset.href.split("/")[-1] if dss_asset else None

                if path and lat is not None and lon is not None:
                    x, y = transformer.transform(lon, lat)
                    storm_info.append({
                        "storm_path": path,
                        "lat": lat,
                        "lon": lon,
                        "x": x,     # SHG projected X
                        "y": y      # SHG projected Y
                    })

        return storm_info
    
    def _get_dss_paths_from_catalog(self) -> List[str]:
        parsed = urlparse(self.catalog_path)
        is_remote = parsed.scheme in ("http", "https", "s3")
        catalog = pystac.Catalog.from_file(self.catalog_path)

        dss_paths = []
        for child_link in catalog.get_links("child"):
            collection = pystac.Collection.from_file(child_link.absolute_href)
            for item in collection.get_all_items():
                item_href = item.self_href
                item_parsed = urlparse(item_href)
                item_dir = PurePosixPath(item_parsed.path).parent
                for asset in item.assets.values():
                    if asset.href.endswith(".dss"):
                        relative_href = asset.href
                        combined = item_dir.joinpath(relative_href)
                        parts = []
                        for part in combined.parts:
                            if part == "..":
                                if parts:
                                    parts.pop()
                            elif part != ".":
                                parts.append(part)
                        normalized_path = PurePosixPath(*parts)
                        if is_remote:
                            bucket = item_parsed.netloc.split(".")[0]
                            dss_url = f"https://{bucket}.s3.amazonaws.com{normalized_path}"
                        else:
                            local_root = Path(item_parsed.path).resolve().parent
                            dss_url = str(local_root.joinpath(relative_href).resolve())
                        dss_paths.append(dss_url)
        return dss_paths

    def _read_dss_as_cumulative_xarray(self, dss_path: str) -> xr.DataArray:
        dss = HecDss(dss_path)
        catalog = dss.get_catalog()
        paths = [p for p in catalog.uncondensed_paths if "SHG" in p and "AORC" in p and "/PRECIPITATION/" in p]
        data_list, time_list = [], []
        sample_grid = None
        cell_size = ll_x = ll_y = None
        
        for path in paths:
            record = dss.get(path)
            grid = record.data.astype(np.float32)
            grid[grid == -3.40282347e38] = np.nan
            data_list.append(grid)
            ts = to_datetime(path.split("/")[4], format="%d%b%Y:%H%M", errors="coerce")
            time_list.append(ts.to_pydatetime())
            if sample_grid is None:
                sample_grid = grid
                cell_size = record.cellSize
                ll_x = record.lowerLeftCellX * cell_size
                ll_y = record.lowerLeftCellY * cell_size

        dss.close()
        rows, cols = sample_grid.shape
        x_coords = ll_x + np.arange(cols) * cell_size
        y_coords = ll_y + np.arange(rows) * cell_size

        data_stack = np.stack(data_list, axis=0)

        return xr.DataArray(
            data_stack,
            dims=("time", "y", "x"),
            coords={
                "time": time_list,
                "y": y_coords,
                "x": x_coords,
            },
            name="precipitation",
            attrs={
                "units": "mm",
                "crs": self.shg_wkt,
                "cell_size_m": cell_size,
            },
        ).sum(dim="time", skipna=True)

    def _process_single_dss(self, dss_path: str) -> Union[xr.DataArray, None]:
        is_remote = urlparse(dss_path).scheme in ("http", "https", "s3")
        storm_id = os.path.basename(dss_path)
        temp_path = None
        try:
            if is_remote:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dss") as tmp:
                    response = requests.get(dss_path, stream=True)
                    response.raise_for_status()
                    with open(tmp.name, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                    temp_path = tmp.name
            else:
                temp_path = dss_path
            da = self._read_dss_as_cumulative_xarray(temp_path)
            return da.expand_dims(storm_path=[storm_id])
        except Exception as e:
            print(f"[ERROR] DSS processing failed for {dss_path}: {e}")
            return None
        finally:
            if is_remote and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _process_all_storms_to_cube(self, max_workers: int = 25) -> xr.DataArray:
        dss_paths = self._get_dss_paths_from_catalog()
        if not dss_paths:
            raise RuntimeError("No DSS paths found in the catalog.")

        all_das = []
        storm_ids = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_dss, path): path for path in dss_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing storms"):
                da = future.result()
                if da is not None:
                    all_das.append(da)
                    storm_ids.append(da.storm_path.values[0])

        if not all_das:
            raise RuntimeError("No DSS files could be successfully processed.")

        combined = xr.concat(all_das, dim="storm_path")
        return combined.assign_coords(storm_path=("storm_path", storm_ids))

    def _read_gdf(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        with open(self.config_path, "r") as f:
            config = json.load(f)
        watershed_path = config["watershed"]["geometry_file"]
        domain_path = config["transposition_region"]["geometry_file"]
        watershed_gdf = gpd.read_file(watershed_path).to_crs(CRS.from_wkt(self.shg_wkt))
        domain_gdf = gpd.read_file(domain_path).to_crs(CRS.from_wkt(self.shg_wkt))
        return watershed_gdf, domain_gdf
    
    @classmethod
    def load(cls, base_path: Union[str, Path], shg_wkt: str = SHG_WKT) -> "Preprocessor":
        base_path = Path(base_path)
        obj = cls("dummy", "dummy", shg_wkt=shg_wkt)
        obj.storm_centers = pd.read_parquet(base_path.with_suffix(".storm_centers.pq"))
        obj.watershed_gdf = gpd.read_file(base_path.with_suffix(".watershed.gpkg"))
        obj.domain_gdf = gpd.read_file(base_path.with_suffix(".domain.gpkg"))
        obj.aggregated_precip = xr.load_dataarray(base_path.with_suffix(".nc"))
        return obj

if __name__ == "__main__":
    catalog_url = "https://trinity-pilot.s3.amazonaws.com/stac/prod-support/storms/catalog.json"
    config_url = "/workspaces/Importance-Sampling-for-SST/data/0_source/geojsons/Trinity/config.json"

    # Instantiate and run preprocessing
    trinity = Preprocessor(catalog_url, config_url, shg_wkt=SHG_WKT)
    trinity.run()
    trinity.save("./data/1_interim/trinity/trinity")