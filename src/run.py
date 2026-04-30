"""
Main entry point for SST Importance Sampling workflow.

Usage:
    python run.py
    python run.py --config /path/to/config.yaml
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Setup Paths
# ----------------------------------------------------------

def setup_paths():
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    data_dir = project_root / "data"

    return {
        "project_root": project_root,
        "config_dir": config_dir,
        "data_dir": data_dir,
    }


# ----------------------------------------------------------
# Config Selection
# ----------------------------------------------------------

def get_config_file(paths, provided_config=None):

    if provided_config:
        config_path = Path(provided_config)
        if config_path.exists():
            logger.info(f"✓ Using provided config: {config_path}")
            return config_path
        logger.error(f"❌ Config file not found: {config_path}")
        return None

    config_dir = paths["config_dir"]

    if not config_dir.exists():
        logger.error(f"❌ Config directory not found: {config_dir}")
        return None

    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    if not yaml_files:
        logger.error("❌ No YAML config files found.")
        return None

    logger.info("\n📂 Available config files:")
    for i, file in enumerate(yaml_files, 1):
        logger.info(f"   {i}. {file.name}")

    while True:
        choice = input("\nSelect config number or enter full path: ").strip()

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(yaml_files):
                return yaml_files[idx]
            logger.error("Invalid selection.")
            continue

        path = Path(choice)
        if path.exists():
            return path

        logger.error("Config file not found.")


# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------

def run_preprocessing(config_file):

    try:
        logger.info("=" * 70)
        logger.info("🚀 Starting Preprocessing")
        logger.info("=" * 70)

        from SSTImportanceSampling import Preprocessor

        logger.info(f"📂 Using config: {config_file}")

        preprocessor = Preprocessor(config_path=str(config_file))
        result = preprocessor.run()

        if result.current_run is None or result.preprocessing is None:
            logger.error("❌ Preprocessing failed.")
            return None

        logger.info("✅ Preprocessing completed successfully!")

        return result.current_run, result.preprocessing

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        logger.exception("Full traceback:")
        return None


# ----------------------------------------------------------
# Sampling
# ----------------------------------------------------------

def run_sampling(config_file, current_run_folder, preprocessing_folder, skip_transpose=False):

    try:
        logger.info("=" * 70)
        logger.info("🎲 Starting Sampling")
        logger.info("=" * 70)

        from SSTImportanceSampling.core.sampler import Sampler

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if "sampling" not in config:
            logger.warning("⚠️ No 'sampling' section found. Skipping sampling.")
            return False

        if skip_transpose:
            logger.info("⚠️ Storm transposition disabled for this run")
            config.setdefault("sampling", {})
            config["sampling"]["apply_transpose"] = False

        sampling_output = Path(current_run_folder) / "sampling"
        sampling_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"   Reading from: {preprocessing_folder}")
        logger.info(f"   Writing to: {sampling_output}")

        sampler = Sampler(
            config=config,
            preprocessor_folder=preprocessing_folder,
            output_folder=sampling_output,
            apply_transpose=config["sampling"].get("apply_transpose", True),
        )

        output_file = sampler.run()

        logger.info("✅ Sampling completed successfully!")
        logger.info(f"📁 Output: {output_file}")

        return True

    except Exception as e:
        logger.error(f"❌ Sampling failed: {e}")
        logger.exception("Full traceback:")
        return False


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SST Importance Sampling Workflow"
    )

    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-transposition", action="store_true")

    return parser.parse_args()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    start_time = datetime.now()
    logger.info(f"⏱️ Started at: {start_time}")

    args = parse_arguments()
    paths = setup_paths()

    config_file = get_config_file(paths, args.config)
    if not config_file:
        return 1

    config_file = Path(config_file).resolve()

    current_run_folder: Optional[Path] = None
    preprocessing_folder: Optional[Path] = None

    # ---------------- Preprocessing ----------------

    if not args.skip_preprocessing:
        logger.info("\n▶️ Running: Preprocessing")

        result = run_preprocessing(config_file)

        if result is None:
            return 1

        current_run_folder, preprocessing_folder = result

    else:
        logger.info("\n⏭️ Skipping preprocessing")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        base_folder = Path(config["output"]["base_folder"]).resolve()

        runs = sorted(
            [d for d in base_folder.iterdir() if (d / "preprocessing").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not runs:
            logger.error("❌ No existing preprocessing found.")
            return 1

        current_run_folder = runs[0]
        preprocessing_folder = current_run_folder / "preprocessing"

        logger.info(f"Using latest preprocessing: {preprocessing_folder}")

    # ---------------- Sampling ----------------

    if not args.skip_sampling:
        logger.info("\n▶️ Running: Sampling")

        success = run_sampling(
            config_file,
            current_run_folder,
            preprocessing_folder,
            skip_transpose=args.skip_transposition,
        )

        if not success:
            return 1

    else:
        logger.info("\n⏭️ Skipping sampling")

    # ---------------- Summary ----------------

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info(f"✨ Workflow completed in {duration:.2f} seconds")
    logger.info(f"📁 Run folder: {current_run_folder}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())