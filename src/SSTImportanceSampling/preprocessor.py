"""SST Importance Sampling preprocessor module."""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from .config import ConfigValidator
from .dss_processor import process_dss_batch
from .fishnet import create_uniform_fishnet_csv
from .geometry import (
    compute_spatial_stats,
    load_and_project_to_shg,
)
from .utils import (
    collect_dss_file_paths,
    get_table_extension,
    read_dataframe,
    resolve_path,
    save_dataframe,
    setup_logger,
    suppress_stdout_stderr,
)
from .validity import generate_valid_storm_placements
from .plots import plot_valid_region_overview

logger = setup_logger(__name__)

# Configuration constants
SUPPORTED_EXPORT_FORMATS = {"csv", "parquet"}
DEFAULT_EXPORT_FORMAT = "csv"

# Required preprocessing files for reuse mode
REQUIRED_FILES = {
    "cumulative_precip": ["cumulative_precip.nc"],
    "storm_centers": ["storm_centers.csv", "storm_centers.parquet"],
    "spatial_bounds": ["spatial_bounds.csv", "spatial_bounds.parquet"],
    "watershed": ["watershed.gpkg"],
    "domain": ["domain.gpkg"],
    "fishnet_points": ["fishnet_points.csv", "fishnet_points.parquet"],
}


class Preprocessor:
    """
    Preprocess geospatial data for SST importance sampling.

    Handles loading geometries, generating fishnet grids, processing DSS files,
    and computing valid storm placements.

    Attributes
    ----------
    config : dict
        Configuration dictionary loaded from YAML.
    project_name : str
        Project identifier.
    run_type : str
        Run type identifier.
    export_format : str
        Output format ('csv' or 'parquet').
    seed : int
        Random seed for reproducibility.
    timestamp : str
        Run timestamp (YYYYMMDD_HHMMSS).
    watershed_stats : pd.Series | None
        Spatial statistics for watershed geometry.
    domain_stats : pd.Series | None
        Spatial statistics for domain geometry.
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Initialize preprocessor with configuration file.

        Parameters
        ----------
        config_path : str | Path
            Path to YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If configuration file does not exist.
        ValueError
            If configuration is invalid.
        """
        config_path = Path(config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config_path = config_path
        self.config_dir = config_path.parent
        self.config = self._load_config()

        # Validate configuration against schema
        self._validate_configuration()

        self.project_name = self.config["project"]["name"]
        self.run_type = self.config["project"]["run_type"]
        self.export_format = self._get_export_format()
        self.base_output = self._setup_output_directory()
        self.seed = self._get_seed()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Statistics storage
        self.watershed_stats: pd.Series | None = None
        self.domain_stats: pd.Series | None = None

        logger.info("Export format: %s", self.export_format.upper())

    def run(self, dry_run: bool = False) -> Path | None:
        """
        Execute the complete preprocessing pipeline.

        Creates a new run folder, handles preprocessing, and saves metadata.

        Parameters
        ----------
        dry_run : bool, optional
            If True, validate configuration but don't process data, by default False

        Returns
        -------
        Path | None
            Path to preprocessing output directory, or None if dry_run=True.
        """
        logger.info("=" * 70)
        logger.info(" SST Importance Sampling - Preprocessor")
        logger.info("=" * 70)

        if dry_run:
            logger.info("🏜️  DRY RUN MODE - Configuration validated, no data processed")
            logger.info("=" * 70)
            return None

        run_folder = self._create_run_folder()
        run_folder.joinpath("sampling").mkdir()

        preprocess_folder = self._handle_preprocessing(run_folder)
        self._save_run_metadata(run_folder, preprocess_folder)

        logger.info("✅ Preprocessing complete: %s", preprocess_folder)
        logger.info("=" * 70)

        return preprocess_folder

    def _validate_configuration(self) -> None:
        """
        Validate configuration against JSON schema.

        Raises
        ------
        ValueError
            If configuration fails validation.
        """
        schema_path = Path(__file__).parent / "schema.json"

        if not schema_path.exists():
            logger.warning("⚠️  Schema file not found, skipping validation")
            return

        validator = ConfigValidator(schema_path)
        is_valid, errors = validator.validate(self.config)

        if not is_valid:
            logger.error("\n" + "=" * 70)
            logger.error("❌ CONFIGURATION VALIDATION FAILED")
            logger.error("=" * 70)
            for error in errors:
                logger.error("  • %s", error)
            logger.error("=" * 70)
            raise ValueError(f"Invalid configuration: {len(errors)} error(s)")

        logger.info("✅ Configuration validation passed")

    def _handle_preprocessing(self, run_folder: Path) -> Path:
        """
        Route preprocessing based on configuration mode.

        Parameters
        ----------
        run_folder : Path
            Current run folder.

        Returns
        -------
        Path
            Preprocessing output directory.

        Raises
        ------
        ValueError
            If mode is invalid or required parameters missing.
        """
        mode = self.config["preprocess"].get("mode", "auto")
        existing_run = self.config["preprocess"].get("existing_run")

        if mode == "force":
            logger.info("Mode: FORCE - Running new preprocessing")
            return self._run_preprocessing(run_folder)

        if mode == "reuse":
            if not existing_run:
                raise ValueError("mode='reuse' requires 'existing_run' parameter")
            logger.info("Mode: REUSE - Using existing preprocessing")
            return self._validate_existing_preprocessing(existing_run)

        if mode == "auto":
            if existing_run:
                logger.info("Mode: AUTO - Existing run found, reusing preprocessing")
                return self._validate_existing_preprocessing(existing_run)
            logger.info("Mode: AUTO - No existing run, running new preprocessing")
            return self._run_preprocessing(run_folder)

        raise ValueError(f"Invalid preprocess mode: {mode}")

    def _run_preprocessing(self, run_folder: Path) -> Path:
        """
        Execute preprocessing pipeline.

        Generates fishnet grid, loads geometries, processes DSS files,
        and computes valid storm placements.

        Parameters
        ----------
        run_folder : Path
            Current run folder.

        Returns
        -------
        Path
            Preprocessing output directory.
        """
        preprocess_folder = run_folder / "preprocessing"
        preprocess_folder.mkdir()

        logger.info("\n" + "-" * 70)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("-" * 70)

        self._process_geometries(preprocess_folder)
        self._process_fishnet(preprocess_folder)
        self._process_dss_files(preprocess_folder)
        self._process_validity_check(preprocess_folder)

        self._print_preprocessing_summary(preprocess_folder)

        logger.info("✅ Preprocessing saved to: %s", preprocess_folder)
        return preprocess_folder

    def _process_geometries(self, output_dir: Path) -> None:
        """
        Load, project, and save geometry files with spatial statistics.

        Saves watershed and domain geometries as GeoPackages and creates
        a comprehensive spatial bounds table with all geometry statistics.

        Parameters
        ----------
        output_dir : Path
            Output directory for geometry files.

        Raises
        ------
        FileNotFoundError
            If input geometry files not found.
        ValueError
            If geometries missing CRS.
        """
        logger.info("\n📍 Processing geometries...")

        watershed_path = self._resolve_config_path("paths", "watershed_geojson")
        domain_path = self._resolve_config_path("paths", "domain_geojson")

        watershed, domain = load_and_project_to_shg(watershed_path, domain_path)

        # Save projected geometries
        watershed.to_file(output_dir / "watershed.gpkg", driver="GPKG")
        domain.to_file(output_dir / "domain.gpkg", driver="GPKG")
        logger.info("  ✓ Saved watershed.gpkg")
        logger.info("  ✓ Saved domain.gpkg")

        # Compute spatial statistics
        self.watershed_stats = compute_spatial_stats(watershed, "watershed")
        self.domain_stats = compute_spatial_stats(domain, "domain")

        # Create spatial bounds table with all statistics
        spatial_bounds_df = pd.DataFrame(
            [self.watershed_stats, self.domain_stats]
        ).reset_index(drop=True)

        # Save spatial bounds as CSV or Parquet
        bounds_path = (
            output_dir
            / f"spatial_bounds.{get_table_extension(self.export_format)}"
        )
        save_dataframe(spatial_bounds_df, bounds_path, self.export_format)
        logger.info(
            "  ✓ Saved spatial_bounds.%s",
            get_table_extension(self.export_format),
        )

        # Log spatial information
        logger.info(
            "  • Watershed centroid: (%.2f, %.2f)",
            self.watershed_stats["centroid_x"],
            self.watershed_stats["centroid_y"],
        )
        logger.info(
            "  • Domain centroid: (%.2f, %.2f)",
            self.domain_stats["centroid_x"],
            self.domain_stats["centroid_y"],
        )
        logger.info(
            "  • Watershed extent: %.2f × %.2f",
            self.watershed_stats["range_x"],
            self.watershed_stats["range_y"],
        )
        logger.info(
            "  • Domain extent: %.2f × %.2f",
            self.domain_stats["range_x"],
            self.domain_stats["range_y"],
        )
        logger.info(
            "  • Watershed area: %.2e sq units",
            self.watershed_stats["area"],
        )
        logger.info(
            "  • Domain area: %.2e sq units",
            self.domain_stats["area"],
        )

    def _process_fishnet(self, output_dir: Path) -> None:
        """
        Generate and save fishnet grid.

        Parameters
        ----------
        output_dir : Path
            Output directory for fishnet file.

        Raises
        ------
        ValueError
            If grid spacing not defined in configuration.
        """
        logger.info("\n🔲 Generating fishnet grid...")

        grid_spacing = self.config["preprocess"].get("grid_spacing")
        if grid_spacing is None:
            raise ValueError("grid_spacing must be defined in configuration")

        fishnet_path = (
            output_dir / f"fishnet_points.{get_table_extension(self.export_format)}"
        )
        create_uniform_fishnet_csv(
            gpkg_path=str(output_dir / "domain.gpkg"),
            spacing=grid_spacing,
            output_path=str(fishnet_path),
        )

        logger.info(
            "  ✓ Saved fishnet_points.%s",
            get_table_extension(self.export_format),
        )
        logger.info("  • Grid spacing: %s units", grid_spacing)

    def _process_dss_files(self, output_dir: Path) -> None:
        """
        Load, process, and save DSS data using optimized parallel processing.

        Extracts cumulative precipitation grids and computes storm centers
        from maximum precipitation cells using parallel file reading
        for maximum performance.

        Key optimizations:
        - Parallel DSS file reading (ProcessPoolExecutor)
        - Process-local maximum computation (cache locality)
        - Streaming netCDF writes
        - Optimized xarray chunking

        Parameters
        ----------
        output_dir : Path
            Output directory for DSS data files.

        Raises
        ------
        ValueError
            If no valid DSS files found.
        Exception
            If DSS processing fails.
        """
        logger.info("\n💧 Processing DSS files...")

        catalog_path = self._resolve_config_path("paths", "dss_catalog")
        dss_files = collect_dss_file_paths(catalog_path)

        if not dss_files:
            logger.warning("  ⚠️  No DSS files found in catalog")
            return

        var_kw = self.config["preprocess"].get(
            "dss_variable_keyword", "PRECIPITATION"
        )
        grid_kw = self.config["preprocess"].get("dss_grid_keyword", "SHG")

        # Get optimized DSS processing parameters
        max_workers = self.config["preprocess"].get("dss_max_workers", 8)

        nc_config = self.config["preprocess"].get("netcdf_compression", {})

        logger.info(f"  • Found {len(dss_files)} DSS files")
        logger.info(f"  • Using {max_workers} parallel workers")

        try:
            # Use optimized parallel processing
            storm_centers, nc_path, centers_path = process_dss_batch(
                dss_files=dss_files,
                output_dir=output_dir,
                export_format=self.export_format,
                max_workers=max_workers,
                compression_config=nc_config,
                var_kw=var_kw,
                grid_kw=grid_kw,
            )

            logger.info(f"  • Processed {len(storm_centers)} storms")

        except Exception as e:
            logger.error(f"Error processing DSS files: {e}")
            raise

    def _process_validity_check(self, output_dir: Path) -> None:
        """
        Execute validity checking for storm placements.

        Determines which grid cells are valid candidates for each storm
        based on transposition constraints.

        Parameters
        ----------
        output_dir : Path
            Output directory for validity results.
        """
        validity_config = self.config["preprocess"].get("validity", {})

        if not validity_config.get("enabled", True):
            logger.info("\n⏭️  Validity checking disabled")
            return

        logger.info("\n✅ Computing valid storm placements...")

        ext = get_table_extension(self.export_format)

        generate_valid_storm_placements(
            fishnet_csv=str(output_dir / f"fishnet_points.{ext}"),
            storm_centers_csv=str(output_dir / f"storm_centers.{ext}"),
            domain_gpkg=str(output_dir / "domain.gpkg"),
            watershed_gpkg=str(output_dir / "watershed.gpkg"),
            output_path=str(output_dir / "valid_placements"),
            export_format=validity_config.get("export_format", self.export_format),
            max_workers=validity_config.get("max_workers", 4),
            fishnet_chunk_size=validity_config.get("fishnet_chunk_size", 50000),
        )

        logger.info(
            "  ✓ Saved valid_placements.%s",
            get_table_extension(self.export_format),
        )

        self._create_plot(output_dir)

    def _print_preprocessing_summary(self, preprocess_folder: Path) -> None:
        """
        Print summary of preprocessing results.

        Parameters
        ----------
        preprocess_folder : Path
            Preprocessing output folder.
        """
        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 70)

        if self.watershed_stats is not None:
            logger.info(
                "Watershed centroid: (%.2f, %.2f)",
                self.watershed_stats["centroid_x"],
                self.watershed_stats["centroid_y"],
            )
        if self.domain_stats is not None:
            logger.info(
                "Domain centroid: (%.2f, %.2f)",
                self.domain_stats["centroid_x"],
                self.domain_stats["centroid_y"],
            )

        # Count storm centers
        for ext in ["parquet", "csv"]:
            candidate = preprocess_folder / f"storm_centers.{ext}"
            if candidate.exists():
                storm_df = read_dataframe(candidate)
                logger.info("Total storms processed: %d", len(storm_df))
                if "pmax_mm" in storm_df.columns:
                    logger.info(
                        "Max precipitation range: %.1f - %.1f mm",
                        storm_df["pmax_mm"].min(),
                        storm_df["pmax_mm"].max(),
                    )
                break

        # Summary of valid placements if available
        for ext in ["parquet", "csv"]:
            candidate = preprocess_folder / f"valid_placements.{ext}"
            if candidate.exists():
                validity_df = read_dataframe(candidate)
                total_candidates = len(validity_df)

                # Get available columns
                columns = list(validity_df.columns)

                logger.info(
                    "Valid placements computed: %d candidate pairs", total_candidates
                )
                logger.info("Validity data columns: %s", ", ".join(columns))
                break

        logger.info("=" * 70)

    def _validate_existing_preprocessing(self, existing_run: str | Path) -> Path:
        """
        Validate and load existing preprocessing folder.

        Parameters
        ----------
        existing_run : str | Path
            Path to existing run directory.

        Returns
        -------
        Path
            Preprocessing directory.

        Raises
        ------
        FileNotFoundError
            If required files are missing.
        """
        folder = resolve_path(existing_run, self.config_dir)
        preprocess_folder = folder / "preprocessing"

        for name, patterns in REQUIRED_FILES.items():
            if not any((preprocess_folder / p).exists() for p in patterns):
                raise FileNotFoundError(
                    f"Missing required file '{name}': {' or '.join(patterns)}"
                )

        # Load existing spatial stats
        bounds_file = None
        for pattern in ["spatial_bounds.parquet", "spatial_bounds.csv"]:
            candidate = preprocess_folder / pattern
            if candidate.exists():
                bounds_file = candidate
                break

        if bounds_file:
            bounds_df = read_dataframe(bounds_file)
            if not bounds_df.empty:
                watershed_row = bounds_df[bounds_df["name"] == "watershed"]
                domain_row = bounds_df[bounds_df["name"] == "domain"]
                if not watershed_row.empty:
                    self.watershed_stats = watershed_row.iloc[0]
                if not domain_row.empty:
                    self.domain_stats = domain_row.iloc[0]

        logger.info("✅ Reusing preprocessing from: %s", preprocess_folder)
        return preprocess_folder

    def _save_run_metadata(self, run_folder: Path, preprocess_folder: Path) -> None:
        """
        Save run configuration and metadata.

        Parameters
        ----------
        run_folder : Path
            Current run folder.
        preprocess_folder : Path
            Preprocessing output folder.
        """
        self.config["run_metadata"] = {
            "timestamp": self.timestamp,
            "seed": self.seed,
            "run_folder": str(run_folder),
            "preprocessing_used": str(preprocess_folder),
            "export_format": self.export_format,
        }

        with open(run_folder / "run_config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(
            "✅ Saved run configuration: %s", run_folder / "run_config.yaml"
        )

    def _load_config(self) -> dict:
        """
        Load YAML configuration file.

        Returns
        -------
        dict
            Configuration dictionary.

        Raises
        ------
        yaml.YAMLError
            If YAML file is invalid.
        """
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_export_format(self) -> str:
        """
        Get and validate export format from configuration.

        Returns
        -------
        str
            Export format ('csv' or 'parquet').

        Raises
        ------
        ValueError
            If format is not supported.
        """
        fmt = (
            self.config["output"]
            .get("export_format", DEFAULT_EXPORT_FORMAT)
            .lower()
        )
        if fmt not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"export_format must be in {SUPPORTED_EXPORT_FORMATS}, got '{fmt}'"
            )
        return fmt

    def _setup_output_directory(self) -> Path:
        """
        Create base output directory from configuration.

        Returns
        -------
        Path
            Base output directory.
        """
        output_dir = resolve_path(
            self.config["output"]["base_folder"], self.config_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_seed(self) -> int:
        """
        Get random seed from configuration or generate new one.

        Returns
        -------
        int
            Random seed value.
        """
        import random

        seed_config = self.config.get("preprocess", {}).get("random_seed")
        return (
            int(seed_config)
            if seed_config is not None
            else random.randint(1000, 9999)
        )

    def _create_run_folder(self) -> Path:
        """
        Create unique run folder with timestamp and seed.

        Returns
        -------
        Path
            New run folder.
        """
        run_folder = (
            self.base_output
            / f"{self.project_name}_{self.run_type}_{self.timestamp}_seed{self.seed}"
        )
        run_folder.mkdir(parents=True, exist_ok=False)
        return run_folder

    def _resolve_config_path(self, *keys: str) -> Path:
        """
        Resolve path from configuration dictionary.

        Parameters
        ----------
        *keys : str
            Nested dictionary keys to traverse.

        Returns
        -------
        Path
            Resolved path.

        Examples
        --------
        >>> self._resolve_config_path("paths", "watershed_geojson")
        """
        value = self.config
        for key in keys:
            value = value[key]
        return resolve_path(value, self.config_dir)

    def _create_plot(self, output_dir: Path) -> None:
        """
        Create visualization plot of valid placement region.

        Parameters
        ----------
        output_dir : Path
            Output directory containing preprocessing data.
        """
        logger.info("\n📊 Creating valid region plot...")

        ext = get_table_extension(self.export_format)

        # Check if required files exist
        valid_placements_path = output_dir / f"valid_placements.{ext}"
        storm_centers_path = output_dir / f"storm_centers.{ext}"

        if not valid_placements_path.exists():
            logger.warning("  ⚠️  Valid placements file not found, skipping plot")
            return

        if not storm_centers_path.exists():
            logger.warning("  ⚠️  Storm centers file not found, skipping plot")
            return

        try:
            plot_valid_region_overview(
                storm_centers_path=storm_centers_path,
                valid_placements_path=valid_placements_path,
                domain_gpkg=output_dir / "domain.gpkg",
                watershed_gpkg=output_dir / "watershed.gpkg",
                output_path=output_dir / "valid_region.png",
                dpi=300,
            )
            logger.info("  ✓ Saved valid_region.png")

        except Exception as e:
            logger.error("  ❌ Error creating plot: %s", str(e))
            logger.debug("Full error:", exc_info=True)
            logger.info("  ⚠️  Continuing preprocessing despite plot creation failure")