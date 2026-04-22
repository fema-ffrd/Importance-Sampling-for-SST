"""Valid storm placement computation for SST Importance Sampling."""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from tqdm import tqdm

from .utils import read_dataframe, save_dataframe


# ==========================================================
# PROCESS SINGLE STORM
# ==========================================================
def _process_single_storm_chunked(args):
    """
    Process storm validity with chunked fishnet (memory efficient).

    Shifts the watershed polygon by the offset between the original storm
    location and each candidate fishnet point, then checks if the shifted
    polygon remains within the domain boundary.

    Parameters
    ----------
    args : tuple
        (storm_row, fish_chunk_x, fish_chunk_y, domain, watershed)
        where fish_chunk_* are numpy arrays of candidate coordinates.

    Returns
    -------
    list
        List of tuples (event_id, x, y) for VALID
        placements only (invalid placements are filtered out).
    """
    storm_row, fish_chunk_x, fish_chunk_y, domain, watershed = args

    event_id = storm_row["event_id"]
    original_x = float(storm_row["x"])
    original_y = float(storm_row["y"])

    offset_x = fish_chunk_x - original_x
    offset_y = fish_chunk_y - original_y

    # Handle both single polygon and multipolygon cases
    if isinstance(watershed, Polygon):
        base_polys = [watershed]
    else:
        base_polys = list(watershed.geoms)

    valid_results = []
    batch_size = 500

    # Process in batches to manage memory
    for batch_start in range(0, len(offset_x), batch_size):
        batch_end = min(batch_start + batch_size, len(offset_x))
        batch_offset_x = offset_x[batch_start:batch_end]
        batch_offset_y = offset_y[batch_start:batch_end]
        batch_fish_x = fish_chunk_x[batch_start:batch_end]
        batch_fish_y = fish_chunk_y[batch_start:batch_end]

        shifted_geoms = []

        # Shift each polygon by each offset
        for poly in base_polys:
            exterior = np.array(poly.exterior.coords)
            holes = (
                [np.array(ring.coords) for ring in poly.interiors]
                if poly.interiors
                else []
            )

            for dx, dy in zip(batch_offset_x, batch_offset_y):
                shifted_ext = exterior - np.array([dx, dy])
                if holes:
                    shifted_holes = [h - np.array([dx, dy]) for h in holes]
                    shifted_geoms.append(Polygon(shifted_ext, shifted_holes))
                else:
                    shifted_geoms.append(Polygon(shifted_ext))

        # Check containment for batch
        if shifted_geoms:
            shifted_geoms_array = np.array(shifted_geoms, dtype=object)
            is_valid = shapely.contains(domain, shifted_geoms_array)

            # Only keep VALID placements
            for x, y, valid in zip(batch_fish_x, batch_fish_y, is_valid):
                if valid:
                    valid_results.append((event_id, float(x), float(y)))

        # Clear memory immediately
        del shifted_geoms, shifted_geoms_array
        gc.collect()

    return valid_results


# ==========================================================
# MAIN: GENERATE VALID STORM PLACEMENTS
# ==========================================================
def generate_valid_storm_placements(
    fishnet_csv: str | Path,
    storm_centers_csv: str | Path,
    domain_gpkg: str | Path,
    watershed_gpkg: str | Path,
    output_path: str | Path,
    export_format: str = "csv",
    max_workers: int = 4,
    fishnet_chunk_size: int = 10000,
) -> Path:
    """
    Generate valid storm placements with memory-efficient parallel processing.

    Determines which fishnet points are valid candidates for each storm
    by checking if the transposed watershed (shifted to each point) remains
    within the domain boundary. Only VALID placements are saved to output.

    Parameters
    ----------
    fishnet_csv : str | Path
        Path to fishnet points file (CSV or Parquet).
        Required columns: x, y

    storm_centers_csv : str | Path
        Path to storm centers file (CSV or Parquet).
        Required columns: event_id, x, y

    domain_gpkg : str | Path
        Path to domain boundary geometry (GeoPackage).

    watershed_gpkg : str | Path
        Path to watershed geometry (GeoPackage).

    output_path : str | Path
        Output file path or directory.
        If directory, saved as "valid_placements.{format}".

    export_format : str, optional
        Output format: "csv" or "parquet", by default "csv".

    max_workers : int, optional
        Number of thread workers for parallel processing, by default 4.

    fishnet_chunk_size : int, optional
        Fishnet chunk size for memory efficiency, by default 10000.

    Returns
    -------
    Path
        Path to output file containing valid placements.

    Raises
    ------
    FileNotFoundError
        If input files don't exist.
    ValueError
        If required columns missing or format unsupported.

    Notes
    -----
    Output file contains only VALID placements with columns:
    - event_id: Storm event identifier
    - x: X coordinate of valid placement
    - y: Y coordinate of valid placement

    A placement is valid if the transposed watershed
    (shifted by offset from original center to candidate point)
    is completely contained within the domain.

    Examples
    --------
    >>> output = generate_valid_storm_placements(
    ...     fishnet_csv="fishnet_points.parquet",
    ...     storm_centers_csv="storm_centers.parquet",
    ...     domain_gpkg="domain.gpkg",
    ...     watershed_gpkg="watershed.gpkg",
    ...     output_path="valid_placements.parquet",
    ...     export_format="parquet"
    ... )
    """
    # Validate export format
    fmt = export_format.lower()
    if fmt not in ["csv", "parquet"]:
        raise ValueError(
            f"export_format must be 'csv' or 'parquet', got '{export_format}'"
        )

    output_path = Path(output_path)

    # Determine output file path
    if output_path.is_dir() or str(output_path).endswith(("\\", "/")):
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"valid_placements.{fmt}"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in [".csv", ".parquet"]:
            output_file = output_path
        else:
            output_file = output_path.parent / f"{output_path.stem}.{fmt}"

    # ========================================
    # Load Data
    # ========================================
    print("Loading data...")
    try:
        fishnet_df = read_dataframe(fishnet_csv)
        storms_df = read_dataframe(storm_centers_csv)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        raise

    # Validate required columns
    if "x" not in fishnet_df.columns or "y" not in fishnet_df.columns:
        raise ValueError("fishnet file must have 'x' and 'y' columns")

    if not all(col in storms_df.columns for col in ["event_id", "x", "y"]):
        raise ValueError("storm_centers file must have 'event_id', 'x', and 'y' columns")

    domain = gpd.read_file(domain_gpkg).geometry.iloc[0]
    watershed = gpd.read_file(watershed_gpkg).geometry.iloc[0]

    fish_x = fishnet_df["x"].values.astype(np.float64)
    fish_y = fishnet_df["y"].values.astype(np.float64)

    print(f"\n{'='*60}")
    print(f"📍 VALIDITY CHECK CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Workers:              {max_workers}")
    print(f"  Storms:               {len(storms_df)}")
    print(f"  Fishnet points:       {len(fish_x):,}")
    print(f"  Total checks:         {len(storms_df) * len(fish_x):,}")
    print(f"  Fishnet chunk size:   {fishnet_chunk_size:,}")
    print(f"  Export format:        {fmt.upper()}")
    print(f"{'='*60}\n")

    # Delete DataFrames after extracting arrays
    del fishnet_df
    gc.collect()

    # Store results in list
    all_valid_results = []

    # Process fishnet in chunks for memory efficiency
    num_chunks = (len(fish_x) + fishnet_chunk_size - 1) // fishnet_chunk_size

    for chunk_idx, chunk_start in enumerate(range(0, len(fish_x), fishnet_chunk_size)):
        chunk_end = min(chunk_start + fishnet_chunk_size, len(fish_x))
        chunk_x = fish_x[chunk_start:chunk_end]
        chunk_y = fish_y[chunk_start:chunk_end]

        print(
            f"📊 Processing chunk {chunk_idx + 1}/{num_chunks} "
            f"({chunk_start:,} - {chunk_end:,})"
        )

        # Re-read storms for this chunk (memory efficient)
        storms_chunk = read_dataframe(storm_centers_csv)

        # Create args for this chunk
        args_list = [
            (storm_row, chunk_x, chunk_y, domain, watershed)
            for _, storm_row in storms_chunk.iterrows()
        ]

        # Process storms for this chunk in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for results in tqdm(
                executor.map(_process_single_storm_chunked, args_list),
                total=len(args_list),
                desc="  Storms",
            ):
                all_valid_results.extend(results)

        # Aggressive cleanup
        del args_list, chunk_x, chunk_y, storms_chunk
        gc.collect()

    # ========================================
    # Export Results
    # ========================================
    print("\nCreating results DataFrame...")
    result_df = pd.DataFrame(
        all_valid_results, columns=["event_id", "x", "y"]
    )

    # Export based on format
    print(f"\nExporting to {fmt.upper()}...")
    save_dataframe(result_df, output_file, fmt)

    # Display summary statistics
    n_valid = len(result_df)
    n_storms = len(storms_df)
    n_candidates = len(fish_x)
    n_total_checks = n_storms * n_candidates

    avg_valid_per_storm = n_valid / n_storms if n_storms > 0 else 0
    validity_pct = (n_valid / n_total_checks * 100) if n_total_checks > 0 else 0

    print(f"\n{'='*60}")
    print(f"📈 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total checks:           {n_total_checks:,}")
    print(f"  Valid placements:       {n_valid:,}")
    print(f"  Validity ratio:         {validity_pct:.2f}%")
    print(f"  Avg valid per storm:    {avg_valid_per_storm:.0f}")
    print(f"{'='*60}\n")

    return output_file