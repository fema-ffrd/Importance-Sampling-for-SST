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
        """
        Initialize PrecipitationTranspose.

        Parameters:
        -----------
        cumulative_precip_file : Path
            Path to cumulative_precip.nc file
        watershed_file : Path
            Path to watershed.gpkg file
        storm_centers_file : Path
            Path to storm_centers.csv file
        """
        self.cumulative_precip_file = Path(cumulative_precip_file)
        self.watershed_file = Path(watershed_file)
        self.storm_centers_file = Path(storm_centers_file)

        self._load_data()

    def _load_data(self):
        """Load all required data for precipitation calculation."""
        logger.info("📂 Loading precipitation data...")

        try:
            # Load cumulative precipitation data
            self.cumulative_precip = xr.load_dataset(self.cumulative_precip_file)["cumulative_precip"]
            logger.info(f"   ✓ Loaded cumulative_precip: shape {self.cumulative_precip.shape}")

            # Load watershed geometry
            self.watershed_gdf = gpd.read_file(self.watershed_file)
            logger.info(f"   ✓ Loaded watershed: {len(self.watershed_gdf)} features")

            # Load storm centers
            self.storm_centers = pd.read_csv(self.storm_centers_file)
            logger.info(f"   ✓ Loaded storm_centers: {len(self.storm_centers)} events")

            # Get transform for rasterio
            self.transform = self.cumulative_precip.rio.transform()
            logger.info(f"   ✓ Extracted raster transform")

        except Exception as e:
            logger.error(f"❌ Error loading precipitation data: {e}")
            raise

    def merge_with_storm_centers(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Merge samples with storm centers and calculate displacement offsets.

        Parameters:
        -----------
        samples : pd.DataFrame
            Sampled events with 'event_id', 'x', 'y' columns

        Returns:
        --------
        pd.DataFrame
            Samples merged with storm center coordinates and offset columns
        """
        logger.info("🔗 Merging samples with storm centers...")

        try:
            # Rename storm center columns to avoid conflicts
            storm_centers_renamed = self.storm_centers.rename(
                columns={'x': 'orig_x', 'y': 'orig_y'}
            )[['event_id', 'orig_x', 'orig_y']]

            # Merge on event_id
            result = samples.merge(storm_centers_renamed, on='event_id', how='left')

            # Calculate offsets
            result['offset_x'] = result['x'] - result['orig_x']
            result['offset_y'] = result['y'] - result['orig_y']

            logger.info(f"   ✓ Merged {len(result)} samples with storm centers")
            logger.info(f"   ✓ Columns: {list(result.columns)}")

            return result

        except Exception as e:
            logger.error(f"❌ Error merging samples with storm centers: {e}")
            raise

    def calculate_watershed_mean_precip(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        For each row, shift watershed by -offset and calculate mean precipitation.

        Parameters:
        -----------
        result : pd.DataFrame
            DataFrame with event_id, x, y, orig_x, orig_y, offset_x, offset_y columns

        Returns:
        --------
        pd.DataFrame
            Input dataframe with added 'precip_mm' column
        """
        logger.info("💧 Calculating watershed mean precipitation...")

        result = result.copy()
        result['precip_mm'] = np.nan

        total_rows = len(result)
        failed_events = []

        for idx, row in result.iterrows():
            if (idx + 1) % max(1, total_rows // 10) == 0:
                logger.info(f"   Progress: {idx + 1}/{total_rows} ({100 * (idx + 1) / total_rows:.1f}%)")

            event_id = str(row['event_id'])
            offset_x = row['offset_x']
            offset_y = row['offset_y']

            try:
                # Get precipitation grid for this event
                precip_grid = self.cumulative_precip.sel(event_id=event_id).values

                # Shift watershed by negative offset
                shifted_watershed = self.watershed_gdf.geometry.translate(-offset_x, -offset_y)

                # Create mask using rasterio geometry_mask
                # transform is ESSENTIAL for proper coordinate transformation
                mask = geometry_mask(
                    geometries=[mapping(geom) for geom in shifted_watershed],
                    out_shape=precip_grid.shape,
                    transform=self.transform,
                    invert=True
                )

                # Extract values within watershed mask
                values = precip_grid[mask]

                if values.size > 0:
                    mean_precip = float(np.nanmean(values))
                    result.loc[idx, 'precip_mm'] = mean_precip
                else:
                    logger.warning(f"   ⚠️  No precipitation data within shifted watershed for event {event_id}")
                    failed_events.append(event_id)

            except Exception as e:
                logger.warning(f"   ⚠️  Error processing event {event_id}: {e}")
                failed_events.append(event_id)

        logger.info(f"   ✓ Calculated precipitation for {(result['precip_mm'].notna().sum())} / {total_rows} samples")

        if failed_events:
            logger.info(f"   ⚠️  Failed to process {len(failed_events)} events")

        return result

    def transpose(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete transposition pipeline:
        1. Merge samples with storm centers
        2. Calculate precipitation statistics

        Parameters:
        -----------
        samples : pd.DataFrame
            Sampled events

        Returns:
        --------
        pd.DataFrame
            Transposed samples with precipitation data
        """
        logger.info("=" * 70)
        logger.info("🌊 Starting Storm Transposition")
        logger.info("=" * 70)

        try:
            # Step 1: Merge with storm centers
            result = self.merge_with_storm_centers(samples)

            # Step 2: Calculate precipitation
            result = self.calculate_watershed_mean_precip(result)

            logger.info("=" * 70)
            logger.info(f"✅ Transposition completed successfully!")
            logger.info(f"📊 Output shape: {result.shape}")
            logger.info(f"📊 Columns: {list(result.columns)}")
            logger.info("=" * 70)

            return result

        except Exception as e:
            logger.error(f"❌ Transposition failed: {e}")
            logger.exception("Full traceback:")
            raise


def create_precipitation_transpose(preprocessor_folder: Path) -> PrecipitationTranspose:
    """
    Factory function to create PrecipitationTranspose with standard file paths.

    Parameters:
    -----------
    preprocessor_folder : Path
        Path to preprocessor output folder

    Returns:
    --------
    PrecipitationTranspose
        Initialized PrecipitationTranspose instance
    """
    preprocessor_folder = Path(preprocessor_folder)

    cumulative_precip_file = preprocessor_folder / "cumulative_precip.nc"
    watershed_file = preprocessor_folder / "watershed.gpkg"
    storm_centers_file = preprocessor_folder / "storm_centers.csv"

    if not cumulative_precip_file.exists():
        raise FileNotFoundError(f"cumulative_precip.nc not found at {cumulative_precip_file}")
    if not watershed_file.exists():
        raise FileNotFoundError(f"watershed.gpkg not found at {watershed_file}")
    if not storm_centers_file.exists():
        raise FileNotFoundError(f"storm_centers.csv not found at {storm_centers_file}")

    return PrecipitationTranspose(cumulative_precip_file, watershed_file, storm_centers_file)