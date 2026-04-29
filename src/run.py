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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup project directories."""
    project_root = Path(__file__).parent.parent  # Go up from src/ to root
    config_dir = project_root / "config"
    data_dir = project_root / "data"

    return {
        "project_root": project_root,
        "config_dir": config_dir,
        "data_dir": data_dir,
    }

def get_config_file(paths, provided_config=None):
    """Get config file path from user or argument."""

    if provided_config:
        config_path = Path(provided_config)
        if config_path.exists():
            logger.info(f"✓ Using provided config: {config_path}")
            return config_path
        else:
            logger.error(f"❌ Config file not found: {config_path}")
            return None

    config_dir = paths["config_dir"]
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        if yaml_files:
            logger.info(f"\n📂 Found config files in {config_dir}:")
            for i, file in enumerate(yaml_files, 1):
                logger.info(f"   {i}. {file.name}")
    else:
        logger.warning(f"⚠️  Config directory not found: {config_dir}")
        yaml_files = []

    while True:
        logger.info("\n🔧 Configuration Selection")
        logger.info("-" * 50)

        if yaml_files:
            logger.info("Enter config file path or number from list above")
            logger.info("(or press Enter to browse manually)")
        else:
            logger.info("Enter full path to config file:")

        user_input = input("Config file: ").strip()

        if not user_input:
            logger.info("Enter the full path to your config file:")
            user_input = input("Path: ").strip()

            if not user_input:
                logger.error("❌ No config file provided")
                return None

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(yaml_files):
                config_path = yaml_files[idx]
                logger.info(f"✓ Selected: {config_path}")
                return config_path
            else:
                logger.error(f"❌ Invalid selection. Please choose 1-{len(yaml_files)}")
                continue

        config_path = Path(user_input)

        if str(config_path).startswith("~"):
            config_path = config_path.expanduser()

        if config_path.exists():
            logger.info(f"✓ Config found: {config_path}")
            return config_path
        else:
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info("Please check the path and try again.\n")

def resolve_path(path_str, config_dir):
    """Resolve path relative to config directory."""
    path = Path(path_str)

    # If absolute path, return as-is
    if path.is_absolute():
        return path

    # If relative path, resolve relative to config directory
    return (config_dir / path).resolve()

def run_preprocessing(config_file):
    """
    Run preprocessing step.

    Returns
    -------
    tuple[Path, Path] | None
        Returns (current_run_folder, preprocessing_folder) from PreprocessingResult
        or None if preprocessing failed.
    """
    try:
        logger.info("=" * 70)
        logger.info("🚀 Starting Preprocessing")
        logger.info("=" * 70)

        from SSTImportanceSampling import Preprocessor

        logger.info(f"📂 Using config: {config_file}")
        preprocessor = Preprocessor(config_path=str(config_file))
        result = preprocessor.run()

        if result.current_run is None or result.preprocessing is None:
            logger.error("❌ Preprocessing returned None paths")
            return None

        logger.info("=" * 70)
        logger.info("✅ Preprocessing completed successfully!")
        logger.info(f"   Current run folder: {result.current_run}")
        logger.info(f"   Preprocessing folder: {result.preprocessing}")
        logger.info("=" * 70)

        return result.current_run, result.preprocessing

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Ensure SSTImportanceSampling is installed: pip install -e .")
        return None
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        logger.exception("Full traceback:")
        return None

def run_sampling(config_file, current_run_folder, preprocessing_folder):
    """
    Run sampling step with storm transposition.

    Parameters
    ----------
    config_file : Path
        Path to configuration file
    current_run_folder : Path
        Current run folder where sampling output will be saved
    preprocessing_folder : Path
        Preprocessing folder containing input data
    """
    try:
        logger.info("=" * 70)
        logger.info("🎲 Starting Sampling with Storm Transposition")
        logger.info("=" * 70)

        from SSTImportanceSampling.core.sampler import Sampler

        logger.info(f"📂 Using config: {config_file}")

        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Get sampling config
        sampling_config = config.get('sampling', {})
        if not sampling_config:
            logger.warning("⚠️  No 'sampling' section in config. Skipping sampling.")
            return False

        # Get random seed
        random_seed = config.get('preprocess', {}).get('random_seed')

        # Get transposition setting (default: True)
        apply_transpose = sampling_config.get('apply_transpose', True)

        logger.info(f"   Storm transposition enabled: {apply_transpose}")

        # Ensure sampling output folder exists
        sampling_output = Path(current_run_folder) / "sampling"
        sampling_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"   Reading preprocessing data from: {preprocessing_folder}")
        logger.info(f"   Writing sampling output to: {sampling_output}")

        # Initialize and run sampler
        sampler = Sampler(
            config=sampling_config,
            preprocessor_folder=str(preprocessing_folder),  # Read from preprocessing/
            output_folder=str(sampling_output),             # Write to current_run/sampling/
            random_seed=random_seed,
            apply_transpose=apply_transpose                 # Enable/disable transposition
        )

        output_file = sampler.run()

        logger.info("=" * 70)
        logger.info("✅ Sampling completed successfully!")
        logger.info(f"📁 Output: {output_file}")
        logger.info("=" * 70)
        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Sampling failed: {e}")
        logger.exception("Full traceback:")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SST Importance Sampling Workflow with Storm Transposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --config config/my_config.yaml
  python run.py -c /full/path/to/config.yaml
  python run.py --skip-preprocessing
  python run.py --skip-sampling
  python run.py --skip-transposition
        """
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file (optional - will prompt if not provided)",
        default=None
    )

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step"
    )

    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip sampling step"
    )

    parser.add_argument(
        "--skip-transposition",
        action="store_true",
        help="Skip storm transposition during sampling"
    )

    return parser.parse_args()

def main():
    """Main workflow."""
    start_time = datetime.now()
    logger.info(f"⏱️  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_arguments()
    paths = setup_paths()
    config_file = get_config_file(paths, args.config)

    if not config_file:
        logger.error("❌ Cannot proceed without config file")
        return 1

    # Resolve config file to absolute path
    config_file = Path(config_file).resolve()
    config_dir = config_file.parent

    current_run_folder: Optional[Path] = None
    preprocessing_folder: Optional[Path] = None

    # Preprocessing
    if not args.skip_preprocessing:
        logger.info(f"\n▶️  Running: Preprocessing")
        preprocessing_result = run_preprocessing(config_file)

        if preprocessing_result is None:
            logger.warning("⚠️  Preprocessing did not complete successfully")
            return 1

        current_run_folder, preprocessing_folder = preprocessing_result
        logger.info(f"✓ Preprocessing result captured")

    else:
        logger.info("\n⏭️  Skipping: Preprocessing")
        logger.info("   Note: Must provide existing preprocessing folder for sampling")

        # Load config to find preprocessing folder
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Try to find existing preprocessing
        output_base = resolve_path(config['output']['base_folder'], config_dir)

        if output_base.exists():
            preprocess_folders = sorted(
                [d for d in output_base.iterdir() if d.is_dir() and (d / "preprocessing").exists()],
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )

            if preprocess_folders:
                current_run_folder = preprocess_folders[0]
                preprocessing_folder = current_run_folder / "preprocessing"
                logger.info(f"   Using existing run folder: {current_run_folder}")
                logger.info(f"   Using existing preprocessing: {preprocessing_folder}")
            else:
                logger.error("❌ No preprocessing folder found. Run preprocessing first.")
                return 1
        else:
            logger.error(f"❌ Output base folder not found: {output_base}")
            return 1

    # Sampling
    if not args.skip_sampling:
        logger.info(f"\n▶️  Running: Sampling")

        if current_run_folder is None or preprocessing_folder is None:
            logger.error("❌ Missing required paths for sampling")
            logger.error("   current_run_folder:", current_run_folder)
            logger.error("   preprocessing_folder:", preprocessing_folder)
            return 1

        # Handle skip-transposition flag
        if args.skip_transposition:
            logger.info("   ⚠️  Storm transposition will be disabled for this run")
            # Modify config to disable transposition
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            if 'sampling' not in config:
                config['sampling'] = {}
            config['sampling']['apply_transpose'] = False

            # Write temporary config or handle it in run_sampling
            logger.info("   Disabling transposition in sampling config")

        run_sampling(config_file, current_run_folder, preprocessing_folder)

    else:
        logger.info("\n⏭️  Skipping: Sampling")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info(f"✨ Workflow completed in {duration:.2f} seconds")

    if current_run_folder:
        logger.info(f"📁 Run folder: {current_run_folder}")
    if preprocessing_folder:
        logger.info(f"📊 Preprocessing data: {preprocessing_folder}")
    if current_run_folder:
        sampling_folder = current_run_folder / "sampling"
        if sampling_folder.exists() and list(sampling_folder.glob("*")):
            logger.info(f"🎲 Sampling output: {sampling_folder}")

    logger.info("=" * 70)

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)