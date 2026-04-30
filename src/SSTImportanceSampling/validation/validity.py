"""
Valid storm placement computation for SST Importance Sampling
Two-stage filtering with interior-point prefilter and split progress bars.
"""

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from ..utils import read_dataframe, save_dataframe

logger = logging.getLogger(__name__)


# ==========================================================
# INTERIOR POINT EXTRACTION
# ==========================================================

def _extract_interior_points(watershed) -> np.ndarray:
    """
    Extract 9 guaranteed interior points using representative_point()
    and shrinking toward bounding box directions.
    """
    rep = watershed.representative_point()
    rx, ry = rep.x, rep.y
    minx, miny, maxx, maxy = watershed.bounds

    directions = [
        (minx, ry),
        (maxx, ry),
        (rx, miny),
        (rx, maxy),
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]

    points = [(rx, ry)]

    for tx, ty in directions:
        shrink_factor = 0.3
        while shrink_factor > 0.01:
            px = rx + shrink_factor * (tx - rx)
            py = ry + shrink_factor * (ty - ry)
            probe = Point(px, py)
            if watershed.contains(probe):
                points.append((px, py))
                break
            shrink_factor *= 0.5

        if len(points) == 9:
            break

    while len(points) < 9:
        points.append(points[-1])

    return np.array(points[:9], dtype=np.float64)


# ==========================================================
# WATERSHED COORD EXTRACTION
# ==========================================================

def _extract_watershed_coords(watershed):
    if isinstance(watershed, Polygon):
        base_polys = [watershed]
    else:
        base_polys = list(watershed.geoms)

    coords_data = []
    for poly in base_polys:
        exterior = np.array(poly.exterior.coords, dtype=np.float64)
        holes = [np.array(ring.coords, dtype=np.float64) for ring in poly.interiors]
        coords_data.append((exterior, holes))

    return coords_data


# ==========================================================
# STAGE 1 — PREFILTER
# ==========================================================

def _prefilter_interior_points(
    interior_points: np.ndarray,
    offset_x: float,
    offset_y: float,
    domain_geom,
) -> bool:
    shifted = interior_points - np.array([offset_x, offset_y])
    for x, y in shifted:
        if not domain_geom.contains(Point(x, y)):
            return False
    return True


# ==========================================================
# STAGE 2 — FULL CONTAINMENT CHECK
# ==========================================================

def _check_full_watershed_containment(
    coords_data,
    offset_x: float,
    offset_y: float,
    domain_geom,
) -> bool:
    for exterior, holes in coords_data:
        shifted_ext = exterior - np.array([offset_x, offset_y])

        if holes:
            shifted_holes = [h - np.array([offset_x, offset_y]) for h in holes]
            shifted_poly = Polygon(shifted_ext, shifted_holes)
        else:
            shifted_poly = Polygon(shifted_ext)

        if not domain_geom.contains(shifted_poly):
            return False

    return True


# ==========================================================
# WORKER FUNCTION
# ==========================================================

def _process_fishnet_chunk(args):
    (fish_chunk_x, fish_chunk_y, storms_data,
     coords_data, interior_points, domain_wkt) = args

    from shapely import wkt
    domain_geom = wkt.loads(domain_wkt)

    stage1_pass_count = 0
    stage2_checks = 0
    results = []

    for storm_path, orig_x, orig_y in storms_data:
        offset_x = fish_chunk_x - orig_x
        offset_y = fish_chunk_y - orig_y

        # Stage 1
        valid_mask = np.array([
            _prefilter_interior_points(interior_points, ox, oy, domain_geom)
            for ox, oy in zip(offset_x, offset_y)
        ])

        candidate_indices = np.where(valid_mask)[0]
        stage1_pass_count += len(candidate_indices)

        # Stage 2
        for idx in candidate_indices:
            stage2_checks += 1
            if _check_full_watershed_containment(
                coords_data, offset_x[idx], offset_y[idx], domain_geom
            ):
                results.append((storm_path, fish_chunk_x[idx], fish_chunk_y[idx]))

    return results, stage1_pass_count, stage2_checks


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def generate_valid_storm_placements(
    fishnet_csv: str | Path,
    storm_centers_csv: str | Path,
    domain_gpkg: str | Path,
    watershed_gpkg: str | Path,
    output_path: str | Path,
    export_format: str = "csv",
    max_workers: int = 4,
    fishnet_chunk_size: int = 50000,
) -> Path:

    fmt = export_format.lower()
    if fmt not in ["csv", "parquet"]:
        raise ValueError("export_format must be 'csv' or 'parquet'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.with_suffix(f".{fmt}")

    print("Loading data...")
    fishnet_df = read_dataframe(fishnet_csv)
    storms_df = read_dataframe(storm_centers_csv)

    domain = gpd.read_file(domain_gpkg).geometry.iloc[0]
    watershed = gpd.read_file(watershed_gpkg).geometry.iloc[0]

    fish_x = fishnet_df["x"].values.astype(np.float64)
    fish_y = fishnet_df["y"].values.astype(np.float64)

    coords_data = _extract_watershed_coords(watershed)
    interior_points = _extract_interior_points(watershed)
    domain_wkt = domain.wkt

    storms_data = storms_df[["storm_path", "x", "y"]].values
    storms_data[:, 1:] = storms_data[:, 1:].astype(np.float64)

    del fishnet_df, storms_df
    gc.collect()

    work_list = []
    for start in range(0, len(fish_x), fishnet_chunk_size):
        end = min(start + fishnet_chunk_size, len(fish_x))
        work_list.append((
            fish_x[start:end],
            fish_y[start:end],
            storms_data,
            coords_data,
            interior_points,
            domain_wkt
        ))

    all_valid_results = []
    total_stage1 = 0
    total_stage2 = 0

    print("\nStage 1: Interior-Point Prefilter")

    with tqdm(total=len(work_list), desc="Stage 1 (Chunks)") as pbar1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_fishnet_chunk, task)
                       for task in work_list]

            for future in as_completed(futures):
                results, stage1_count, stage2_count = future.result()

                all_valid_results.extend(results)
                total_stage1 += stage1_count
                total_stage2 += stage2_count

                pbar1.update(1)

    print(f"\n✅ Stage 1 complete.")
    print(f"   Candidates passed prefilter: {total_stage1:,}")

    print("\nStage 2: Full Geometry Checks")
    with tqdm(total=total_stage2,
              desc="Stage 2 (Detailed Checks)") as pbar2:
        pbar2.update(total_stage2)

    print(f"\n✅ Stage 2 complete.")
    print(f"   Full geometry checks performed: {total_stage2:,}")
    print(f"   Valid placements: {len(all_valid_results):,}")

    result_df = pd.DataFrame(
        all_valid_results,
        columns=["storm_path", "x", "y"]
    )

    print("\nSaving results...")
    save_dataframe(result_df, output_file, fmt)

    return output_file