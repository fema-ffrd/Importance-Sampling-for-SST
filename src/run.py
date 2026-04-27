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
    
    # If config provided via command line, use it
    if provided_config:
        config_path = Path(provided_config)
        if config_path.exists():
            logger.info(f"✓ Using provided config: {config_path}")
            return config_path
        else:
            logger.error(f"❌ Config file not found: {config_path}")
            return None
    
    # Show available configs in config directory
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
    
    # Interactive selection
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
            # Manual entry
            logger.info("Enter the full path to your config file:")
            user_input = input("Path: ").strip()
            
            if not user_input:
                logger.error("❌ No config file provided")
                return None
        
        # Check if it's a number (selection from list)
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(yaml_files):
                config_path = yaml_files[idx]
                logger.info(f"✓ Selected: {config_path}")
                return config_path
            else:
                logger.error(f"❌ Invalid selection. Please choose 1-{len(yaml_files)}")
                continue
        
        # Direct path provided
        config_path = Path(user_input)
        
        # Expand user home directory if needed
        if str(config_path).startswith("~"):
            config_path = config_path.expanduser()
        
        if config_path.exists():
            logger.info(f"✓ Config found: {config_path}")
            return config_path
        else:
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info("Please check the path and try again.\n")


def run_preprocessing(config_file):
    """Run preprocessing step."""
    try:
        logger.info("=" * 70)
        logger.info("🚀 Starting SST Importance Sampling Preprocessing")
        logger.info("=" * 70)
        
        from SSTImportanceSampling import Preprocessor
        
        logger.info(f"📂 Using config: {config_file}")
        
        # Initialize preprocessor with config
        preprocessor = Preprocessor(config_path=str(config_file))
        
        # Run preprocessing
        preprocessor.run()
        
        logger.info("=" * 70)
        logger.info("✅ Preprocessing completed successfully!")
        logger.info("=" * 70)
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Ensure SSTImportanceSampling is installed: pip install -e .")
        return False
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        logger.exception("Full traceback:")
        return False


def run_analysis():
    """Run analysis step (future)."""
    logger.info("=" * 70)
    logger.info("🔬 Analysis step - Not yet implemented")
    logger.info("=" * 70)
    return True


def run_postprocessing():
    """Run postprocessing step (future)."""
    logger.info("=" * 70)
    logger.info("📊 Postprocessing step - Not yet implemented")
    logger.info("=" * 70)
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SST Importance Sampling Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --config config/my_config.yaml
  python run.py -c /full/path/to/config.yaml
  python run.py --skip-analysis --skip-postprocessing
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
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step"
    )
    
    parser.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Skip postprocessing step"
    )
    
    return parser.parse_args()


def main():
    """Main workflow."""
    start_time = datetime.now()
    logger.info(f"⏱️  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup paths
    paths = setup_paths()
    
    # Get config file (interactive or from argument)
    config_file = get_config_file(paths, args.config)
    
    if not config_file:
        logger.error("❌ Cannot proceed without config file")
        return 1
    
    # Run workflow steps
    steps = [
        ("Preprocessing", lambda: run_preprocessing(config_file), args.skip_preprocessing),
        ("Analysis", run_analysis, args.skip_analysis),
        ("Postprocessing", run_postprocessing, args.skip_postprocessing),
    ]
    
    success = True
    for step_name, step_func, skip in steps:
        if skip:
            logger.info(f"\n⏭️  Skipping: {step_name}")
            continue
        
        logger.info(f"\n▶️  Running: {step_name}")
        if not step_func():
            logger.warning(f"⚠️  {step_name} did not complete successfully")
            # Uncomment to stop on error:
            # success = False
            # break
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✨ Workflow completed in {duration:.2f} seconds")
    logger.info(f"📁 Output saved to: {paths['data_dir']}")
    logger.info("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)