import os
import xarray as xr
import geopandas as gpd
import s3fs
import pandas as pd
import rioxarray  # Needed to activate .rio
from shapely.geometry import shape
import scipy

MM_TO_INCH = 0.0393701
AORC_VAR = "APCP_surface"
S3_BUCKET = "noaa-nws-aorc-v1-1-1km"


class AORCEventGenerator:
    """
    A class to generate storm events from AORC gridded precipitation data by:
    - Downloading and clipping the dataset
    - Decluster peak rainfall times
    - Saving events as NetCDF files
    """

    def __init__(
        self,
        geojson_path,
        start_date,
        end_date,
        duration_hours=72,
        threshold_mm=63.5,
        min_separation_hours=120,
        max_events_per_year=10,
        output_dir="events"
    ):
        self.geojson_path = geojson_path
        self.start_date = start_date
        self.end_date = end_date
        self.duration_hours = duration_hours
        self.threshold_mm = threshold_mm
        self.min_separation_hours = min_separation_hours
        self.max_events_per_year = max_events_per_year
        self.output_dir = output_dir

        self.ds = None
        self.event_times = []

    def load_and_clip_dataset(self):
        """Load AORC data from S3, subset in time and space, and clip to watershed."""
        s3 = s3fs.S3FileSystem(anon=True)
        years = list(range(self.start_date.year, self.end_date.year + 1))
        datasets = []

        for year in years:
            s3_map = s3fs.S3Map(root=f"{S3_BUCKET}/{year}.zarr", s3=s3, check=False)
            ds_year = xr.open_dataset(s3_map, engine="zarr", consolidated=True, chunks="auto")
            datasets.append(ds_year)

        ds = xr.concat(datasets, dim="time")
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        domain = gpd.read_file(self.geojson_path).to_crs("EPSG:4326")
        bounds = shape(domain.geometry.iloc[0]).bounds

        ds = ds.sel(
            longitude=slice(bounds[0], bounds[2]),
            latitude=slice(bounds[1], bounds[3])
        )

        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        self.ds = ds.rio.clip(domain.geometry, domain.crs, drop=True, all_touched=True)

    def decluster_events(self):
        """Decluster rolling rainfall sums to find independent storm events."""
        precip = self.ds[AORC_VAR]
        rolling_sum = precip.rolling(time=self.duration_hours).sum()
        max_over_area = rolling_sum.max(dim=["latitude", "longitude"])

        df = max_over_area.to_dataframe(name="max_precip").dropna()
        df = df[df.max_precip > self.threshold_mm].sort_values("max_precip", ascending=False)
        df["year"] = df.index.year

        selected_times = []
        blocked_times = set()

        for year, group in df.groupby("year"):
            year_selected = []
            for t in group.index:
                if t in blocked_times:
                    continue
                year_selected.append(t)
                block_start = t - pd.Timedelta(hours=self.min_separation_hours)
                block_end = t + pd.Timedelta(hours=self.min_separation_hours)
                blocked_times.update(pd.date_range(block_start, block_end, freq="1H"))

                if len(year_selected) >= self.max_events_per_year:
                    break

            selected_times.extend(year_selected)

        self.event_times = sorted(selected_times)

    def save_events(self):
        """Save selected storm events to NetCDF files."""
        os.makedirs(self.output_dir, exist_ok=True)

        for i, t in enumerate(self.event_times):
            start = pd.to_datetime(t)
            end = start + pd.Timedelta(hours=self.duration_hours - 1)
            event_ds = self.ds.sel(time=slice(start, end))
            filename = f"{self.output_dir}/event_{i+1}_{start.strftime('%Y%m%dT%H')}.nc"
            event_ds.to_netcdf(filename)
            print(f"Saved event to {filename}")

    def run_all(self):
        """Run the full workflow: load data → decluster → save events."""
        self.load_and_clip_dataset()
        self.decluster_events()
        self.save_events()
