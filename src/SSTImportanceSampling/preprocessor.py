# preprocessor.py

import yaml
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import xarray as xr

from .utils import collect_dss_file_paths, suppress_stdout_stderr
from .geometry import load_and_project_to_shg, compute_centroids
from .dss_reader import read_dss_cumulative


class Preprocessor:

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(self, config_path: str):

        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.project_name = self.config["project"]["name"]

        self.run_type = self.config["project"]["run_type"]

        self.base_output = self._resolve_path(
            self.config["output"]["base_folder"]
        )
        self.base_output.mkdir(parents=True, exist_ok=True)

        self.seed = self._initialize_seed()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================
    # PUBLIC ENTRY POINT
    # ==========================================================
    def run(self):

        print("\n==========================================")
        print(" SST Importance Sampling - Preprocessor ")
        print("==========================================\n")

        # ✅ Always create a new run folder
        run_folder = (
            self.base_output
            / f"{self.project_name}_{self.run_type}_{self.timestamp}_seed{self.seed}"
        )
        run_folder.mkdir(parents=True, exist_ok=False)

        sampling_folder = run_folder / "sampling"
        sampling_folder.mkdir()

        # ✅ Handle preprocessing
        preprocess_folder = self._handle_preprocessing(run_folder)

        # ✅ Save run metadata
        self._save_run_metadata(run_folder, preprocess_folder)

        print(f"\n✅ Using preprocessing from:\n{preprocess_folder}")
        print(f"✅ Current run folder:\n{run_folder}\n")

        return preprocess_folder

    # ==========================================================
    # PREPROCESS MODE HANDLER
    # ==========================================================
    def _handle_preprocessing(self, run_folder: Path) -> Path:

        mode = self.config["preprocess"].get("mode", "auto")
        existing_run = self.config["preprocess"].get("existing_run")

        if mode == "force":
            return self._run_preprocessing(run_folder)

        if mode == "reuse":
            if not existing_run:
                raise ValueError("mode='reuse' requires existing_run.")
            return self._validate_existing(existing_run)

        if mode == "auto":
            if existing_run:
                return self._validate_existing(existing_run)
            return self._run_preprocessing(run_folder)

        raise ValueError("Invalid preprocess mode.")

    # ==========================================================
    # RUN NEW PREPROCESSING
    # ==========================================================
    def _run_preprocessing(self, run_folder: Path) -> Path:

        print("Running preprocessing...\n")

        preprocess_folder = run_folder / "preprocessing"
        preprocess_folder.mkdir()

        # ------------------------------------------------------
        # 1. Load & project geometries
        # ------------------------------------------------------
        watershed_path = self._resolve_path(
            self.config["paths"]["watershed_geojson"]
        )
        domain_path = self._resolve_path(
            self.config["paths"]["domain_geojson"]
        )

        watershed, domain = load_and_project_to_shg(
            watershed_path,
            domain_path
        )

        watershed.to_file(
            preprocess_folder / "watershed_projected.gpkg",
            driver="GPKG"
        )
        domain.to_file(
            preprocess_folder / "domain_projected.gpkg",
            driver="GPKG"
        )

        w_centroid, d_centroid = compute_centroids(watershed, domain)

        centers_df = pd.DataFrame([
            {"name": "watershed", "x": w_centroid.x, "y": w_centroid.y},
            {"name": "domain", "x": d_centroid.x, "y": d_centroid.y},
        ])
        centers_df.to_csv(
            preprocess_folder / "geometry_centers.csv",
            index=False
        )

        # ------------------------------------------------------
        # 2. DSS Processing
        # ------------------------------------------------------
        catalog_path = self._resolve_path(
            self.config["paths"]["dss_catalog"]
        )

        dss_files = collect_dss_file_paths(catalog_path)

        cumulative_list = []
        storm_centers = []

        var_kw = self.config["preprocess"].get(
            "dss_variable_keyword", "PRECIPITATION"
        )
        grid_kw = self.config["preprocess"].get(
            "dss_grid_keyword", "SHG"
        )

        print("\nProcessing DSS files...\n")

        for dss_file in dss_files:
            event_name = Path(dss_file).stem
            print(f"  → {event_name}")

            with suppress_stdout_stderr():
                da = read_dss_cumulative(
                    dss_file,
                    variable_keyword=var_kw,
                    grid_keyword=grid_kw,
                )

            cumulative_list.append(da)

            flat = da.values.argmax()
            i, j = divmod(flat, da.shape[1])

            storm_centers.append({
                "event": event_name,
                "x": float(da.x.values[j]),
                "y": float(da.y.values[i])
            })

        stacked = xr.concat(
            cumulative_list,
            dim=xr.DataArray(
                [Path(f).stem for f in dss_files],
                dims="event",
                name="event"
            )
        )

        stacked.name = "cumulative_precip"

        nc_opts = self.config["preprocess"].get("netcdf_compression", {})
        encoding = {
            "cumulative_precip": {
                "zlib": nc_opts.get("zlib", True),
                "complevel": nc_opts.get("complevel", 4)
            }
        }

        stacked.to_netcdf(
            preprocess_folder / "cumulative_precip.nc",
            encoding=encoding
        )

        pd.DataFrame(storm_centers).to_csv(
            preprocess_folder / "storm_centers.csv",
            index=False
        )

        print(f"\n✅ Preprocessing saved to: {preprocess_folder}\n")

        return preprocess_folder

    # ==========================================================
    # REUSE EXISTING PREPROCESSING
    # ==========================================================
    def _validate_existing(self, existing_run: str) -> Path:

        folder = self._resolve_path(existing_run)
        preprocess_folder = folder / "preprocessing"

        required = [
            "cumulative_precip.nc",
            "storm_centers.csv",
            "watershed_projected.gpkg",
            "domain_projected.gpkg",
        ]

        for file in required:
            if not (preprocess_folder / file).exists():
                raise FileNotFoundError(
                    f"Missing required file: {file}"
                )

        print("Reusing existing preprocessing.\n")
        return preprocess_folder

    # ==========================================================
    # SAVE RUN METADATA
    # ==========================================================
    def _save_run_metadata(self, run_folder: Path, preprocess_folder: Path):

        self.config["run_metadata"] = {
            "timestamp": self.timestamp,
            "seed": self.seed,
            "run_folder": str(run_folder),
            "preprocessing_used": str(preprocess_folder),
        }

        with open(run_folder / "run_config.yaml", "w") as f:
            yaml.dump(self.config, f)

    # ==========================================================
    # UTILITIES
    # ==========================================================
    def _resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        return (self.config_dir / p).resolve()

    def _initialize_seed(self) -> int:
        seed = self.config.get("preprocess", {}).get("random_seed")
        return int(seed) if seed is not None else random.randint(1000, 9999)