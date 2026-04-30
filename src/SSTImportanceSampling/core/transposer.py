"""
Transpose module for precipitation calculations and storm transposition.
Handles shifting watersheds and calculating mean precipitation within shifted geometries.
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
from pathlib import Path
from shapely.geometry import mapping
from rasterio.features import geometry_mask

logger = logging.getLogger(__name__)


class PrecipitationTranspose:
    """
    Calculate precipitation statistics for transposed storms.
    Handles watershed shifting and mean precipitation calculation.
    """

    def __init__(self, cumulative_precip_file: Path, watershed_file: Path, storm_centers_file: Path):
        self.cumulative_precip_file = Path(cumulative_precip_file)
        self.watershed_file = Path(watershed_file)
        self.storm_centers_file = Path(storm_centers_file)

        self._load_data()

    def _load_data(self):
        """Load all required data for precipitation calculation."""
        logger.info("📂 Loading precipitation data...")

        try:
            self.cumulative_precip = xr.load_dataset(
                self.cumulative_precip_file
            )["cumulative_precip"]

            self.watershed_gdf = gpd.read_file(self.watershed_file)
            self.storm_centers = pd.read_csv(self.storm_centers_file)

            self.transform = self.cumulative_precip.rio.transform()

            logger.info(f"   ✓ cumulative_precip shape: {self.cumulative_precip.shape}")
            logger.info(f"   ✓ watershed features: {len(self.watershed_gdf)}")
            logger.info(f"   ✓ storm centers: {len(self.storm_centers)}")

        except Exception as e:
            logger.error(f"❌ Error loading precipitation data: {e}")
            raise

    def merge_with_storm_centers(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Merge samples with storm centers."""
        logger.info("🔗 Merging samples with storm centers...")

        storm_centers_renamed = self.storm_centers.rename(
            columns={"x": "orig_x", "y": "orig_y"}
        )[["storm_path", "orig_x", "orig_y"]]

        result = samples.merge(storm_centers_renamed, on="storm_path", how="left")

        logger.info(f"   ✓ Merged {len(result)} samples")

        return result

    def calculate_watershed_mean_precip(self, result: pd.DataFrame) -> pd.DataFrame:
        """Shift watershed and calculate mean precipitation."""
        logger.info("💧 Calculating watershed mean precipitation...")

        result = result.copy()
        result["precip"] = np.nan

        total_rows = len(result)
        failed_events = []

        for idx, row in result.iterrows():

            if (idx + 1) % max(1, total_rows // 10) == 0:
                logger.info(
                    f"   Progress: {idx + 1}/{total_rows} "
                    f"({100 * (idx + 1) / total_rows:.1f}%)"
                )

            storm_path = str(row["storm_path"])

            offset_x = row["x"] - row["orig_x"]
            offset_y = row["y"] - row["orig_y"]

            try:
                precip_grid = self.cumulative_precip.sel(
                    storm_path=storm_path
                ).values

                shifted_watershed = self.watershed_gdf.geometry.translate(
                    -offset_x, -offset_y
                )

                mask = geometry_mask(
                    geometries=[mapping(geom) for geom in shifted_watershed],
                    out_shape=precip_grid.shape,
                    transform=self.transform,
                    invert=True,
                )

                values = precip_grid[mask]

                if values.size > 0:
                    result.loc[idx, "precip"] = float(np.nanmean(values))
                else:
                    failed_events.append(storm_path)

            except Exception as e:
                logger.warning(f"   ⚠️ Error processing {storm_path}: {e}")
                failed_events.append(storm_path)

        logger.info(
            f"   ✓ Calculated precipitation for "
            f"{result['precip'].notna().sum()} / {total_rows} samples"
        )

        if failed_events:
            logger.info(f"   ⚠️ Failed events: {len(failed_events)}")

        return result

    def transpose(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete transposition pipeline.
        """
        logger.info("=" * 70)
        logger.info("🌊 Starting Storm Transposition")
        logger.info("=" * 70)

        result = self.merge_with_storm_centers(samples)
        result = self.calculate_watershed_mean_precip(result)

        # ✅ Reorder columns
        desired_order = [
            "storm_path",
            "orig_x",
            "orig_y",
            "x",
            "y",
            "weight",
            "precip",
        ]

        result = result[desired_order]

        # ✅ Add event_number as first column (1 to n)
        result.insert(0, "event_number", range(1, len(result) + 1))

        logger.info("=" * 70)
        logger.info("✅ Transposition completed successfully!")
        logger.info(f"📊 Output shape: {result.shape}")
        logger.info(f"📊 Columns: {list(result.columns)}")
        logger.info("=" * 70)

        return result


def create_precipitation_transpose(preprocessor_folder: Path) -> PrecipitationTranspose:
    """
    Factory function to create PrecipitationTranspose with standard file paths.
    """

    preprocessor_folder = Path(preprocessor_folder)

    cumulative_precip_file = preprocessor_folder / "cumulative_precip.nc"
    watershed_file = preprocessor_folder / "watershed.gpkg"
    storm_centers_file = preprocessor_folder / "storm_centers.csv"

    if not cumulative_precip_file.exists():
        raise FileNotFoundError(f"Missing: {cumulative_precip_file}")
    if not watershed_file.exists():
        raise FileNotFoundError(f"Missing: {watershed_file}")
    if not storm_centers_file.exists():
        raise FileNotFoundError(f"Missing: {storm_centers_file}")

    return PrecipitationTranspose(
        cumulative_precip_file,
        watershed_file,
        storm_centers_file,
    )