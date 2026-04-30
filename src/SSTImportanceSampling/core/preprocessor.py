"""SST Importance Sampling preprocessor module."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml
import ulid

from ..config import ConfigValidator
from ..io.dss_processor import process_dss_batch
from ..geometry.fishnet import create_uniform_fishnet_csv
from ..geometry.geometry import (
    compute_spatial_stats,
    load_and_project_to_shg,
)
from ..utils import (
    collect_dss_file_paths,
    get_table_extension,
    read_dataframe,
    resolve_path,
    save_dataframe,
    setup_logger,
)
from ..validation.validity import generate_valid_storm_placements
from ..visualization.plots import plot_valid_region_overview

logger = setup_logger(__name__)

SUPPORTED_EXPORT_FORMATS = {"csv", "parquet"}
DEFAULT_EXPORT_FORMAT = "csv"

REQUIRED_FILES = {
    "cumulative_precip": ["cumulative_precip.nc"],
    "storm_centers": ["storm_centers.csv", "storm_centers.parquet"],
    "spatial_bounds": ["spatial_bounds.csv", "spatial_bounds.parquet"],
    "watershed": ["watershed.gpkg"],
    "domain": ["domain.gpkg"],
    "fishnet_points": ["fishnet_points.csv", "fishnet_points.parquet"],
}


class PreprocessingResult(NamedTuple):
    current_run: Path | None
    preprocessing: Path | None


class Preprocessor:
    def __init__(self, config_path: str | Path) -> None:

        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config_path = config_path
        self.config_dir = config_path.parent
        self.config = self._load_config()

        self._validate_configuration()

        self.project_name = self.config["project"]["name"]
        self.run_type = self.config["project"]["run_type"]
        self.export_format = self._get_export_format()
        self.base_output = self._setup_output_directory()
        self.seed = self._get_seed()

        # ✅ Production-grade run identifiers
        self.run_id = str(ulid.new())
        self.timestamp_utc = datetime.now(timezone.utc).isoformat()

        self.watershed_stats: pd.Series | None = None
        self.domain_stats: pd.Series | None = None

        logger.info("Run ID: %s", self.run_id)
        logger.info("Export format: %s", self.export_format.upper())

    # ======================================================
    # MAIN ENTRY
    # ======================================================

    def run(self, dry_run: bool = False) -> PreprocessingResult:

        logger.info("=" * 70)
        logger.info(" SST Importance Sampling - Preprocessor")
        logger.info("=" * 70)

        if dry_run:
            logger.info("🏜️  DRY RUN MODE - Configuration validated, no data processed")
            logger.info("=" * 70)
            return PreprocessingResult(None, None)

        current_run_folder = self._create_run_folder()
        current_run_folder.joinpath("sampling").mkdir()

        preprocessing_folder = self._handle_preprocessing(current_run_folder)
        self._save_run_metadata(current_run_folder, preprocessing_folder)

        logger.info("✅ Preprocessing complete: %s", preprocessing_folder)
        logger.info("=" * 70)

        return PreprocessingResult(current_run_folder, preprocessing_folder)

    # ======================================================
    # VALIDATION
    # ======================================================

    def _validate_configuration(self) -> None:

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

    # ======================================================
    # PREPROCESS ROUTING
    # ======================================================

    def _handle_preprocessing(self, current_run_folder: Path) -> Path:

        mode = self.config["preprocess"].get("mode", "auto")
        existing_run = self.config["preprocess"].get("existing_run")

        if mode == "force":
            logger.info("Mode: FORCE - Running new preprocessing")
            return self._run_preprocessing(current_run_folder)

        if mode == "reuse":
            if not existing_run:
                raise ValueError("mode='reuse' requires 'existing_run'")
            logger.info("Mode: REUSE - Using existing preprocessing")
            return self._validate_existing_preprocessing(existing_run)

        if mode == "auto":
            if existing_run:
                logger.info("Mode: AUTO - Reusing existing preprocessing")
                return self._validate_existing_preprocessing(existing_run)
            logger.info("Mode: AUTO - Running new preprocessing")
            return self._run_preprocessing(current_run_folder)

        raise ValueError(f"Invalid preprocess mode: {mode}")

    # ======================================================
    # PIPELINE
    # ======================================================

    def _run_preprocessing(self, current_run_folder: Path) -> Path:

        preprocessing_folder = current_run_folder / "preprocessing"
        preprocessing_folder.mkdir()

        logger.info("\n" + "-" * 70)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("-" * 70)

        self._process_geometries(preprocessing_folder)
        self._process_fishnet(preprocessing_folder)
        self._process_dss_files(preprocessing_folder)
        self._process_validity_check(preprocessing_folder)
        self._print_preprocessing_summary(preprocessing_folder)

        logger.info("✅ Preprocessing saved to: %s", preprocessing_folder)
        return preprocessing_folder

    # ======================================================
    # GEOMETRY
    # ======================================================

    def _process_geometries(self, output_dir: Path) -> None:

        logger.info("\n📍 Processing geometries...")

        watershed_path = self._resolve_config_path("paths", "watershed_geojson")
        domain_path = self._resolve_config_path("paths", "domain_geojson")

        watershed, domain = load_and_project_to_shg(watershed_path, domain_path)

        watershed.to_file(output_dir / "watershed.gpkg", driver="GPKG")
        domain.to_file(output_dir / "domain.gpkg", driver="GPKG")

        self.watershed_stats = compute_spatial_stats(watershed, "watershed")
        self.domain_stats = compute_spatial_stats(domain, "domain")

        spatial_bounds_df = pd.DataFrame(
            [self.watershed_stats, self.domain_stats]
        ).reset_index(drop=True)

        bounds_path = (
            output_dir
            / f"spatial_bounds.{get_table_extension(self.export_format)}"
        )
        save_dataframe(spatial_bounds_df, bounds_path, self.export_format)

    # ======================================================
    # FISHNET
    # ======================================================

    def _process_fishnet(self, output_dir: Path) -> None:

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

        logger.info("  ✓ Saved fishnet_points.%s", get_table_extension(self.export_format))
        logger.info("  • Grid spacing: %s units", grid_spacing)

    # ======================================================
    # DSS
    # ======================================================

    def _process_dss_files(self, output_dir: Path) -> None:

        logger.info("\n💧 Processing DSS files...")

        catalog_path = self._resolve_config_path("paths", "dss_catalog")
        dss_files = collect_dss_file_paths(catalog_path)

        if not dss_files:
            logger.warning("No DSS files found")
            return

        var_kw = self.config["preprocess"].get("dss_variable_keyword", "PRECIPITATION")
        grid_kw = self.config["preprocess"].get("dss_grid_keyword", "SHG")
        max_workers = self.config["preprocess"].get("dss_max_workers", 8)
        nc_config = self.config["preprocess"].get("netcdf_compression", {})

        storm_centers, _, _ = process_dss_batch(
            dss_files=dss_files,
            output_dir=output_dir,
            export_format=self.export_format,
            max_workers=max_workers,
            compression_config=nc_config,
            var_kw=var_kw,
            grid_kw=grid_kw,
        )

        logger.info("Processed %d storms", len(storm_centers))

    # ======================================================
    # VALIDITY
    # ======================================================

    def _process_validity_check(self, output_dir: Path) -> None:

        validity_config = self.config["preprocess"].get("validity", {})

        if not validity_config.get("enabled", True):
            logger.info("Validity checking disabled")
            return

        ext = get_table_extension(self.export_format)

        generate_valid_storm_placements(
            fishnet_csv=str(output_dir / f"fishnet_points.{ext}"),
            storm_centers_csv=str(output_dir / f"storm_centers.{ext}"),
            domain_gpkg=str(output_dir / "domain.gpkg"),
            watershed_gpkg=str(output_dir / "watershed.gpkg"),
            output_path=str(output_dir / "valid_placements"),
            export_format=self.export_format,
            max_workers=validity_config.get("max_workers", 4),
            fishnet_chunk_size=validity_config.get("fishnet_chunk_size", 50000),
        )

        self._create_plot(output_dir)

    # ======================================================
    # SUMMARY
    # ======================================================

    def _print_preprocessing_summary(self, preprocessing_folder: Path) -> None:

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

        for ext in ["parquet", "csv"]:
            candidate = preprocessing_folder / f"storm_centers.{ext}"
            if candidate.exists():
                storm_df = read_dataframe(candidate)
                logger.info("Total storms processed: %d", len(storm_df))
                break

        logger.info("=" * 70)

    # ======================================================
    # EXISTING PREPROCESS
    # ======================================================

    def _validate_existing_preprocessing(self, existing_run: str | Path) -> Path:

        folder = resolve_path(existing_run, self.config_dir)
        preprocessing_folder = folder / "preprocessing"

        for name, patterns in REQUIRED_FILES.items():
            if not any((preprocessing_folder / p).exists() for p in patterns):
                raise FileNotFoundError(
                    f"Missing required file '{name}': {' or '.join(patterns)}"
                )

        return preprocessing_folder

    # ======================================================
    # FOLDER CREATION
    # ======================================================

    def _create_run_folder(self) -> Path:
        run_folder = (
            self.base_output
            / f"{self.project_name}-{self.run_type}-{self.run_id}"
        )
        run_folder.mkdir(parents=True, exist_ok=False)
        return run_folder

    # ======================================================
    # METADATA
    # ======================================================

    def _save_run_metadata(
        self,
        current_run_folder: Path,
        preprocessing_folder: Path,
    ) -> None:

        self.config["run_metadata"] = {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "seed": self.seed,
            "run_folder": str(current_run_folder),
            "preprocessing_used": str(preprocessing_folder),
            "export_format": self.export_format,
        }

        with open(current_run_folder / "run_config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    # ======================================================
    # UTILS
    # ======================================================

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_export_format(self) -> str:
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
        output_dir = resolve_path(
            self.config["output"]["base_folder"], self.config_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_seed(self) -> int:
        seed_config = self.config.get("global", {}).get("random_seed")
        if seed_config is None:
            raise ValueError("global.random_seed must be defined in config")
        return int(seed_config)

    def _resolve_config_path(self, *keys: str) -> Path:
        value = self.config
        for key in keys:
            value = value[key]
        return resolve_path(value, self.config_dir)

    # ======================================================
    # PLOT
    # ======================================================

    def _create_plot(self, output_dir: Path) -> None:

        ext = get_table_extension(self.export_format)

        valid_placements_path = output_dir / f"valid_placements.{ext}"
        storm_centers_path = output_dir / f"storm_centers.{ext}"

        if not valid_placements_path.exists() or not storm_centers_path.exists():
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
        except Exception:
            logger.warning("Plot creation failed (continuing).")