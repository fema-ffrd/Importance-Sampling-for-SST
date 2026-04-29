"""Optimized DSS file reader for HEC-DSS files."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from hecdss import HecDss

logger = logging.getLogger(__name__)


def read_dss_cumulative(
    dss_path: str | Path,
    variable_keyword: str = "PRECIPITATION",
    grid_keyword: str = "SHG",
) -> xr.DataArray:
    """
    Read cumulative precipitation grid from HEC-DSS file.

    Combines all matching precipitation grids in the file and returns
    their cumulative sum as an xarray DataArray.

    Parameters
    ----------
    dss_path : str | Path
        Path to HEC-DSS file.
    variable_keyword : str, optional
        Keyword to filter variable names (default: "PRECIPITATION").
    grid_keyword : str, optional
        Keyword to filter grid names (default: "SHG").

    Returns
    -------
    xr.DataArray
        Cumulative precipitation grid with dimensions (y, x) and
        coordinates (x, y) in the same projection as the DSS file.
        Attributes include cell_size and source_file.

    Raises
    ------
    ValueError
        If no matching grids found in the DSS file.
    IOError
        If DSS file cannot be opened or read.

    Notes
    -----
    - Missing values (HEC-DSS convention: -3.4028235e38) are converted to NaN
    - All grids are stacked and summed along the time dimension
    - Coordinates are computed from lower-left cell position and cell size
    - Data type is preserved as float32 for memory efficiency

    Examples
    --------
    >>> data = read_dss_cumulative(
    ...     "storm_precip.dss",
    ...     variable_keyword="PRECIPITATION",
    ...     grid_keyword="SHG"
    ... )
    >>> print(data.shape)  # (rows, cols)
    >>> print(data.attrs)  # metadata
    """
    dss_path = Path(dss_path)

    try:
        dss = HecDss(str(dss_path))
    except Exception as e:
        raise IOError(f"Cannot open DSS file {dss_path}: {e}")

    try:
        catalog = dss.get_catalog()
    except Exception as e:
        dss.close()
        raise ValueError(f"Cannot read catalog from {dss_path}: {e}")

    # Filter paths by keywords
    paths = [
        p
        for p in catalog.uncondensed_paths
        if variable_keyword in p and grid_keyword in p
    ]

    if not paths:
        dss.close()
        raise ValueError(
            f"No grids matching '{variable_keyword}' and '{grid_keyword}' in {dss_path}"
        )

    logger.debug(f"Found {len(paths)} matching grids in {dss_path}")

    data_list = []
    cell_size = None
    ll_x = None
    ll_y = None

    # Read all grids
    for path in paths:
        try:
            record = dss.get(path)
            grid = record.data.astype(np.float32)

            # HEC-DSS missing value convention
            grid[grid == -3.4028235e38] = np.nan

            data_list.append(grid)

            # Extract grid metadata from first record
            if cell_size is None:
                cell_size = record.cellSize
                ll_x = record.lowerLeftCellX * cell_size
                ll_y = record.lowerLeftCellY * cell_size

        except Exception as e:
            logger.warning(f"Error reading grid {path}: {e}")
            continue

    dss.close()

    if not data_list:
        raise ValueError(f"No valid grids could be read from {dss_path}")

    # Stack and sum grids
    stack = np.stack(data_list, axis=0)
    cumulative = np.nansum(stack, axis=0)

    # Compute coordinates
    rows, cols = cumulative.shape
    x = ll_x + (0.5 + np.arange(cols)) * cell_size
    y = ll_y + (0.5 + np.arange(rows)) * cell_size

    # Create xarray DataArray with metadata
    return xr.DataArray(
        cumulative,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        attrs={
            "cell_size": float(cell_size),
            "source_file": str(dss_path),
            "num_grids_stacked": len(data_list),
            "variable_keyword": variable_keyword,
            "grid_keyword": grid_keyword,
        },
    )


def validate_dss_file(dss_path: str | Path) -> Tuple[bool, str]:
    """
    Validate that a DSS file can be opened and contains expected grids.

    Parameters
    ----------
    dss_path : str | Path
        Path to HEC-DSS file.

    Returns
    -------
    tuple
        (is_valid, message) - Boolean validity and descriptive message.

    Examples
    --------
    >>> is_valid, msg = validate_dss_file("test.dss")
    >>> if is_valid:
    ...     print("File is valid:", msg)
    """
    dss_path = Path(dss_path)

    if not dss_path.exists():
        return False, f"File not found: {dss_path}"

    try:
        dss = HecDss(str(dss_path))
        catalog = dss.get_catalog()
        num_records = len(catalog.uncondensed_paths)
        dss.close()

        if num_records == 0:
            return False, "No records found in DSS file"

        return True, f"Valid DSS file with {num_records} records"

    except Exception as e:
        return False, f"Error reading DSS file: {e}"