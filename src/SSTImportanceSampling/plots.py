"""Plotting utilities for SST Importance Sampling."""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from .utils import read_dataframe

logger = logging.getLogger(__name__)


def plot_valid_region_overview(
    storm_centers_path: Path,
    valid_placements_path: Path,
    domain_gpkg: Path,
    watershed_gpkg: Path,
    output_path: Path,
    dpi: int = 300,
    figsize: tuple = (14, 10),
) -> None:
    """
    Create overview plot of watershed, domain, valid placements, and storm centers.

    Parameters
    ----------
    storm_centers_path : Path
        Path to storm_centers.csv or .parquet
    valid_placements_path : Path
        Path to valid_placements.csv or .parquet
    domain_gpkg : Path
        Path to domain.gpkg
    watershed_gpkg : Path
        Path to watershed.gpkg
    output_path : Path
        Output path for saved figure
    dpi : int, optional
        Resolution in dots per inch (default: 300)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (14, 10))
    """

    logger.info("Creating valid region overview plot...")

    # Validate input files
    for path, name in [
        (storm_centers_path, "storm centers"),
        (valid_placements_path, "valid placements"),
        (domain_gpkg, "domain"),
        (watershed_gpkg, "watershed"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")

    # Read the files
    logger.info("Loading geospatial data...")
    watershed_gdf = gpd.read_file(watershed_gpkg)
    domain_gdf = gpd.read_file(domain_gpkg)

    logger.info("Loading storm centers...")
    storm_centers_df = read_dataframe(storm_centers_path)
    storm_centers_gdf = gpd.GeoDataFrame(
        storm_centers_df,
        geometry=gpd.points_from_xy(storm_centers_df['x'], storm_centers_df['y']),
        crs=watershed_gdf.crs,
    )

    logger.info("Loading valid placements...")
    valid_placements_df = read_dataframe(valid_placements_path)
    valid_placements_gdf = gpd.GeoDataFrame(
        valid_placements_df,
        geometry=gpd.points_from_xy(
            valid_placements_df['x'],
            valid_placements_df['y']
        ),
        crs=watershed_gdf.crs,
    )

    # Plot both on same axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot watershed (blue)
    watershed_gdf.plot(
        ax=ax,
        alpha=0.2,
        facecolor='lightblue',
        edgecolor='blue',
        linewidth=2,
        label="Watershed",
    )

    # Plot domain (red)
    domain_gdf.plot(
        ax=ax,
        alpha=0.2,
        facecolor='lightcoral',
        edgecolor='red',
        linewidth=2,
        label="Domain",
    )

    # Plot storm centers (orange)
    storm_centers_gdf.plot(
        ax=ax,
        color='orange',
        markersize=120,
        marker='o',
        edgecolor='darkorange',
        linewidth=1,
        label=f"Storm Centers (n={len(storm_centers_gdf)})",
        zorder=5,
    )

    # Plot valid placements (green)
    valid_placements_gdf.plot(
        ax=ax,
        color='green',
        markersize=10,
        alpha=0.3,
        label=f"Valid Placements (n={len(valid_placements_gdf):,})",
        zorder=2,
    )

    # Set axis limits to watershed + domain bounds
    ws_bounds = watershed_gdf.total_bounds
    dom_bounds = domain_gdf.total_bounds

    combined_bounds = np.array([
        min(ws_bounds[0], dom_bounds[0]),
        min(ws_bounds[1], dom_bounds[1]),
        max(ws_bounds[2], dom_bounds[2]),
        max(ws_bounds[3], dom_bounds[3]),
    ])

    minx, miny, maxx, maxy = combined_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_title("Valid Storm Center Placements", fontsize=16, fontweight="bold")
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=12)

    plt.tight_layout()

    # Save and show
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving plot to: %s", output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("✅ Plot saved: %s", output_path)