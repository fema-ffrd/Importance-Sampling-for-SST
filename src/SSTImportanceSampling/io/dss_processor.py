"""Optimized DSS file processing with parallel I/O and streaming."""

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ==========================================================
# PARALLEL DSS READING WITH MAX COMPUTATION
# ==========================================================

def _read_dss_and_extract_max(args: Tuple[str, str, str]) -> Tuple[str, xr.DataArray, dict]:
    """
    Read DSS file and extract storm center in single operation.

    Returns
    -------
    tuple
        (storm_path, data_array, storm_center_dict)
    """
    dss_file, var_kw, grid_kw = args

    from .dss_reader import read_dss_cumulative
    from ..utils import suppress_stdout_stderr

    storm_path = Path(dss_file).stem

    try:
        with suppress_stdout_stderr():
            data_array = read_dss_cumulative(
                dss_file,
                variable_keyword=var_kw,
                grid_keyword=grid_kw,
            )

        # Find max location (no longer storing max value)
        idx_max = int(data_array.values.argmax())
        row, col = divmod(idx_max, int(data_array.shape[1]))

        storm_center = {
            "storm_path": storm_path,
            "x": float(data_array.x.values[col]),
            "y": float(data_array.y.values[row]),
        }

        return storm_path, data_array, storm_center

    except Exception as e:
        logger.error(f"Error processing {dss_file}: {e}")
        raise


# ==========================================================
# OPTIMIZED BATCH PROCESSING
# ==========================================================

def process_dss_batch(
    dss_files: list,
    output_dir: Path,
    export_format: str = "csv",
    max_workers: int = 8,
    compression_config: dict = None,
    var_kw: str = "PRECIPITATION",
    grid_kw: str = "SHG",
) -> Tuple[list, Path, Path]:

    if compression_config is None:
        compression_config = {"zlib": True, "complevel": 4}

    if not dss_files:
        raise ValueError("No DSS files provided")

    storm_centers = []
    data_arrays = []
    storm_paths = []
    nc_path = output_dir / "cumulative_precip.nc"

    logger.info(f"Processing {len(dss_files)} DSS files with {max_workers} workers...")

    args_list = [(f, var_kw, grid_kw) for f in dss_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_read_dss_and_extract_max, args): args[0]
            for args in args_list
        }

        with tqdm(
            total=len(dss_files),
            desc="  Processing DSS",
            unit=" file",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            ncols=80,
        ) as pbar:
            for future in as_completed(futures):
                try:
                    storm_path, data_array, storm_center = future.result()

                    storm_centers.append(storm_center)
                    data_arrays.append(data_array)
                    storm_paths.append(storm_path)

                    pbar.update(1)

                except Exception as e:
                    dss_file = futures[future]
                    logger.error(f"Error processing {dss_file}: {e}")
                    pbar.update(1)

    if not data_arrays:
        raise ValueError("No valid DSS data extracted")

    logger.info(f"Successfully processed {len(storm_centers)} storms")

    # Sort consistently by storm_path
    sorted_indices = np.argsort(storm_paths)
    storm_centers = [storm_centers[i] for i in sorted_indices]
    data_arrays = [data_arrays[i] for i in sorted_indices]
    storm_paths = [storm_paths[i] for i in sorted_indices]

    logger.info("Stacking data arrays and creating netCDF...")
    _create_optimized_netcdf(
        data_arrays=data_arrays,
        storm_paths=storm_paths,
        output_path=nc_path,
        compression_config=compression_config,
    )

    # Save storm centers
    from ..utils import save_dataframe, get_table_extension

    centers_path = output_dir / f"storm_centers.{get_table_extension(export_format)}"
    save_dataframe(pd.DataFrame(storm_centers), centers_path, export_format)

    logger.info(f"✓ Saved {centers_path.name}")
    logger.info(f"✓ Saved {nc_path.name}")

    return storm_centers, nc_path, centers_path


# ==========================================================
# OPTIMIZED NETCDF STACKING
# ==========================================================

def _create_optimized_netcdf(
    data_arrays: list,
    storm_paths: list,
    output_path: Path,
    compression_config: dict = None,
) -> None:

    if compression_config is None:
        compression_config = {"zlib": True, "complevel": 4}

    if not data_arrays:
        raise ValueError("No data arrays to write")

    logger.info(f"Stacking {len(data_arrays)} data arrays...")

    stacked = xr.concat(
        data_arrays,
        dim=xr.DataArray(storm_paths, dims="storm_path", name="storm_path"),
    )
    stacked.name = "cumulative_precip"

    n_storms = len(storm_paths)
    spatial_shape = data_arrays[0].shape

    optimal_chunks = {
        "storm_path": min(50, n_storms),
        str(data_arrays[0].dims[0]): spatial_shape[0],
        str(data_arrays[0].dims[1]): spatial_shape[1],
    }

    logger.info(f"Applying optimal chunking: {optimal_chunks}")

    try:
        import dask.array as da  # noqa

        stacked = stacked.chunk(optimal_chunks)
        logger.debug("Using dask for lazy evaluation")
    except ImportError:
        logger.warning("dask not available, using default chunking")
    except Exception as e:
        logger.warning(f"Could not apply dask chunking: {e}")

    encoding = {
        "cumulative_precip": {
            "zlib": compression_config.get("zlib", True),
            "complevel": compression_config.get("complevel", 4),
        }
    }

    logger.info(f"Writing {len(storm_paths)} storms to {output_path}")
    stacked.to_netcdf(output_path, encoding=encoding, mode="w")
    logger.info(f"✓ netCDF created: {output_path}")

    del stacked, data_arrays
    gc.collect()