# dss_reader.py

import numpy as np
import xarray as xr
from hecdss import HecDss


def read_dss_cumulative(dss_path, variable_keyword="PRECIPITATION", grid_keyword="SHG"):
    dss = HecDss(dss_path)
    catalog = dss.get_catalog()

    paths = [
        p for p in catalog.uncondensed_paths
        if variable_keyword in p and grid_keyword in p
    ]

    if not paths:
        raise ValueError(f"No matching grids in {dss_path}")

    data_list = []
    cell_size = None
    ll_x = None
    ll_y = None

    for path in paths:
        record = dss.get(path)
        grid = record.data.astype(np.float32)
        grid[grid == -3.4028235e38] = np.nan
        data_list.append(grid)

        if cell_size is None:
            cell_size = record.cellSize
            ll_x = record.lowerLeftCellX * cell_size
            ll_y = record.lowerLeftCellY * cell_size

    dss.close()

    stack = np.stack(data_list)
    cumulative = np.nansum(stack, axis=0)

    rows, cols = cumulative.shape
    x = ll_x + (0.5 + np.arange(cols)) * cell_size
    y = ll_y + (0.5 + np.arange(rows)) * cell_size

    return xr.DataArray(
        cumulative,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        attrs={"cell_size": cell_size}
    )