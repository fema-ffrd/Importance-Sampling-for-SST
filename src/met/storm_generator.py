import os
import datetime
import logging
import xarray as xr
import geopandas as gpd
import s3fs
import pandas as pd
from shapely.geometry import shape

MM_TO_INCH = 0.0393701
AORC_VAR = "APCP_surface"
S3_BUCKET = "noaa-nws-aorc-v1-1-1km"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class StormCatalogConfig:
    def __init__(self,
                 geojson_path: str,
                 start_date: datetime.datetime,
                 end_date: datetime.datetime,
                 duration_hours: int = 72,
                 threshold_mm: float = 63.5,
                 min_separation_hours: int = 120,
                 max_events_per_year: int = 10,
                 output_dir: str = "events"):
        self.geojson_path = geojson_path
        self.start_date = start_date
        self.end_date = end_date
        self.duration_hours = duration_hours
        self.threshold_mm = threshold_mm
        self.min_separation_hours = min_separation_hours
        self.max_events_per_year = max_events_per_year
        self.output_dir = output_dir


class StormCatalogGenerator:
    def __init__(self, config: StormCatalogConfig):
        self.config = config

    def open_dataset(self):
        logging.info("Opening AORC data from S3...")
        s3 = s3fs.S3FileSystem(anon=True)
        years = list(range(self.config.start_date.year, self.config.end_date.year + 1))
        datasets = []

        for year in years:
            s3_map = s3fs.S3Map(root=f"{S3_BUCKET}/{year}.zarr", s3=s3, check=False)
            ds_year = xr.open_dataset(s3_map, engine="zarr", consolidated=True, chunks="auto")
            datasets.append(ds_year)

        ds = xr.concat(datasets, dim="time")
        ds = ds.sel(time=slice(self.config.start_date, self.config.end_date))

        domain = gpd.read_file(self.config.geojson_path).to_crs("EPSG:4326")
        bounds = shape(domain.geometry.iloc[0]).bounds

        ds = ds.sel(
            longitude=slice(bounds[0], bounds[2]),
            latitude=slice(bounds[1], bounds[3])
        )

        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        ds_clipped = ds.rio.clip(domain.geometry, domain.crs, drop=True, all_touched=True)

        return ds_clipped

    def decluster_events(self, ds):
        logging.info("Declustering heavy precipitation events...")
        precip = ds[AORC_VAR]
        rolling_sum = precip.rolling(time=self.config.duration_hours, center=False).sum()
        max_over_area = rolling_sum.max(dim=["latitude", "longitude"])

        df = max_over_area.to_dataframe(name="max_precip").dropna()
        df = df[df.max_precip > self.config.threshold_mm]
        df = df.sort_values("max_precip", ascending=False)
        df["year"] = df.index.year

        selected_times = []
        blocked_times = set()

        for year, group in df.groupby("year"):
            year_selected = []
            for t in group.index:
                if t in blocked_times:
                    continue
                year_selected.append(t)

                block_start = t - pd.Timedelta(hours=self.config.min_separation_hours)
                block_end = t + pd.Timedelta(hours=self.config.min_separation_hours)
                blocked_times.update(pd.date_range(block_start, block_end, freq='1H'))

                if len(year_selected) >= self.config.max_events_per_year:
                    break
            selected_times.extend(year_selected)

        logging.info(f"Found {len(selected_times)} declustered events.")
        return sorted(selected_times)

    def save_events(self, ds, event_times):
        logging.info("Saving events to NetCDF...")
        os.makedirs(self.config.output_dir, exist_ok=True)

        for i, t in enumerate(event_times):
            start = pd.to_datetime(t)
            end = start + pd.Timedelta(hours=self.config.duration_hours - 1)
            event_ds = ds.sel(time=slice(start, end))
            file_path = f"{self.config.output_dir}/event_{i+1}_{start.strftime('%Y%m%dT%H')}.nc"
            event_ds.to_netcdf(file_path)
            logging.info(f"Saved event to {file_path}")
