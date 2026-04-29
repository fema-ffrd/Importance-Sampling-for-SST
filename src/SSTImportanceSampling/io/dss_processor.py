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

    Runs in separate process to avoid GIL and I/O blocking.
    Computes maximum precipitation location on the same process
    that read the data (better cache locality).

    Parameters
    ----------
    args : tuple
        (dss_file_path, variable_keyword, grid_keyword)

    Returns
    -------
    tuple
        (event_id, data_array, storm_center_dict)

    Raises
    ------
    Exception
        If DSS file cannot be read or is invalid.
    """
    dss_file, var_kw, grid_kw = args

    from .dss_reader import read_dss_cumulative
    from ..utils import suppress_stdout_stderr

    event_id = Path(dss_file).stem

    try:
        with suppress_stdout_stderr():
            data_array = read_dss_cumulative(
                dss_file,
                variable_keyword=var_kw,
                grid_keyword=grid_kw,
            )

        # Find max on the process that loaded the data (better cache locality)
        idx_max = int(data_array.values.argmax())
        row, col = divmod(idx_max, int(data_array.shape[1]))
        pmax = float(data_array.values[row, col])

        storm_center = {
            "event_id": event_id,
            "x": float(data_array.x.values[col]),
            "y": float(data_array.y.values[row]),
            "pmax_mm": pmax,
        }

        return event_id, data_array, storm_center

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
    """
    Process batch of DSS files in parallel with streaming output.

    Uses ProcessPoolExecutor for true parallelization across multiple CPU cores.
    Results are collected and sorted to maintain consistency.

    Parameters
    ----------
    dss_files : list
        List of DSS file paths.
    output_dir : Path
        Output directory for results.
    export_format : str, optional
        Output format ('csv' or 'parquet'), by default "csv".
    max_workers : int, optional
        Number of parallel processes, by default 8.
    compression_config : dict, optional
        NetCDF compression settings (zlib, complevel).
    var_kw : str, optional
        DSS variable keyword, by default "PRECIPITATION".
    grid_kw : str, optional
        DSS grid keyword, by default "SHG".

    Returns
    -------
    tuple
        (storm_centers_list, nc_path, centers_path)

    Raises
    ------
    ValueError
        If no valid DSS files found or processing fails.

    Notes
    -----
    Memory usage is approximately:
        max_workers * max_file_size
    
    Example
    -------
    >>> storm_centers, nc_path, centers_path = process_dss_batch(
    ...     dss_files=["file1.dss", "file2.dss"],
    ...     output_dir=Path("./output"),
    ...     max_workers=8
    ... )
    """
    if compression_config is None:
        compression_config = {"zlib": True, "complevel": 4}

    if not dss_files:
        raise ValueError("No DSS files provided")

    storm_centers = []
    data_arrays = []
    event_ids = []
    nc_path = output_dir / "cumulative_precip.nc"

    logger.info(f"Processing {len(dss_files)} DSS files with {max_workers} workers...")

    # Prepare arguments for parallel processing
    args_list = [(f, var_kw, grid_kw) for f in dss_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_read_dss_and_extract_max, args): args[0] 
            for args in args_list
        }

        # Process results as they complete (not in order)
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
                    event_id, data_array, storm_center = future.result()

                    storm_centers.append(storm_center)
                    data_arrays.append(data_array)
                    event_ids.append(event_id)

                    pbar.update(1)

                except Exception as e:
                    dss_file = futures[future]
                    logger.error(f"Error processing {dss_file}: {e}")
                    pbar.update(1)

    if not data_arrays:
        raise ValueError("No valid DSS data extracted")

    logger.info(f"Successfully processed {len(storm_centers)} storms")

    # Sort by event_id to maintain consistency
    sorted_indices = np.argsort(event_ids)
    storm_centers = [storm_centers[i] for i in sorted_indices]
    data_arrays = [data_arrays[i] for i in sorted_indices]
    event_ids = [event_ids[i] for i in sorted_indices]

    # Create optimized netCDF
    logger.info("Stacking data arrays and creating netCDF...")
    _create_optimized_netcdf(
        data_arrays=data_arrays,
        event_ids=event_ids,
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
    event_ids: list,
    output_path: Path,
    compression_config: dict = None,
) -> None:
    """
    Create netCDF with optimized chunking and compression.

    Uses intelligent chunk sizing for efficient reads and writes.
    Attempts to use dask for lazy evaluation on large stacks.

    Parameters
    ----------
    data_arrays : list
        List of xarray DataArrays.
    event_ids : list
        Event identifiers (in order).
    output_path : Path
        Output netCDF file path.
    compression_config : dict, optional
        Compression settings (zlib, complevel).

    Raises
    ------
    ValueError
        If data_arrays is empty.
    Exception
        If netCDF creation fails.

    Notes
    -----
    Optimal chunking is:
        - event_id: 50 (read 50 events at a time)
        - spatial dims: full spatial extent (for 2D operations)
    
    This balances memory usage with I/O efficiency.
    """
    if compression_config is None:
        compression_config = {"zlib": True, "complevel": 4}

    if not data_arrays:
        raise ValueError("No data arrays to write")

    logger.info(f"Stacking {len(data_arrays)} data arrays...")

    # Stack all arrays
    stacked = xr.concat(
        data_arrays,
        dim=xr.DataArray(event_ids, dims="event_id", name="event_id"),
    )
    stacked.name = "cumulative_precip"

    # Calculate optimal chunk sizes
    n_events = len(event_ids)
    spatial_shape = data_arrays[0].shape

    # Optimize for typical access patterns (by event)
    optimal_chunks = {
        "event_id": min(50, n_events),  # Read 50 events at a time
        str(data_arrays[0].dims[0]): spatial_shape[0],  # Full y dimension
        str(data_arrays[0].dims[1]): spatial_shape[1],  # Full x dimension
    }

    logger.info(f"Applying optimal chunking: {optimal_chunks}")

    # Try to use dask for lazy evaluation
    try:
        import dask.array as da

        stacked = stacked.chunk(optimal_chunks)
        logger.debug("Using dask for lazy evaluation")
    except ImportError:
        logger.warning("dask not available, using default chunking")
    except Exception as e:
        logger.warning(f"Could not apply dask chunking: {e}, continuing without dask")

    encoding = {
        "cumulative_precip": {
            "zlib": compression_config.get("zlib", True),
            "complevel": compression_config.get("complevel", 4),
        }
    }

    logger.info(f"Writing {len(event_ids)} events to {output_path}")
    stacked.to_netcdf(output_path, encoding=encoding, mode="w")
    logger.info(f"✓ netCDF created: {output_path}")

    # Cleanup
    del stacked, data_arrays
    gc.collect()