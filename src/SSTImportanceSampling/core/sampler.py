"""
Uniform Sampling for Storm Transposition.
Integrates precipitation calculation into sampling workflow.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from .transposer import create_precipitation_transpose

logger = logging.getLogger(__name__)


def sample_storms_uniformly(df, n_samples=10, seed=None):
    """
    Sample n events uniformly with replacement.
    For each event, sample one storm center uniformly from available rows.

    Parameters:
    -----------
    df : DataFrame with 'event_id' column and other storm center data (x, y coords)
    n_samples : number of times to repeat the sampling
    seed : random seed for reproducibility

    Returns:
    --------
    DataFrame with sampled events and uniform weights (1/n_samples)
    """
    if seed is not None:
        np.random.seed(seed)

    results = []
    weight = 1.0 / n_samples  # Uniform weight for all samples

    for i in range(n_samples):
        # Step 1: Sample one unique event_id uniformly
        unique_events = df['event_id'].unique()
        sampled_event = np.random.choice(unique_events)

        # Step 2: Get all rows with this event_id
        event_rows = df[df['event_id'] == sampled_event]

        # Step 3: Sample ONE row uniformly from this event (storm center)
        sampled_row = event_rows.sample(n=1, replace=False).copy()

        # Step 4: Add weight column (1/n)
        sampled_row['weight'] = weight

        results.append(sampled_row)

    return pd.concat(results, ignore_index=True)


class Sampler:
    """
    Uniform Sampler for storm events.
    Reads preprocessor outputs, performs uniform sampling, and calculates precipitation statistics.
    """

    def __init__(
        self,
        config: dict,
        preprocessor_folder: Path,
        output_folder: Path,
        random_seed: int = None,
        apply_transpose: bool = True
    ):
        """
        Initialize Sampler.

        Parameters:
        -----------
        config : dict
            Sampling configuration from config.yaml
        preprocessor_folder : Path
            Path to preprocessor output folder (where to READ data from: preprocessing/)
        output_folder : Path
            Path where sampling results will be saved (sampling/ subfolder, NOT preprocessing/)
        random_seed : int, optional
            Random seed for reproducibility
        apply_transpose : bool, optional
            Whether to apply transposition and calculate precipitation statistics (default: True)
        """
        self.config = config
        self.preprocessor_folder = Path(preprocessor_folder)
        self.output_folder = Path(output_folder)
        self.random_seed = random_seed
        self.apply_transpose = apply_transpose
        self.precip_transpose = None

        logger.info(f"📊 Initializing Sampler")
        logger.info(f"   Events: {self.config['n_events']}")
        logger.info(f"   Reading from: {self.preprocessor_folder}")
        logger.info(f"   Saving to: {self.output_folder}")
        logger.info(f"   Apply transposition: {self.apply_transpose}")

        # Initialize precipitation transpose if needed
        if self.apply_transpose:
            try:
                self.precip_transpose = create_precipitation_transpose(self.preprocessor_folder)
            except FileNotFoundError as e:
                logger.warning(f"⚠️  Transposition files not found: {e}")
                logger.warning(f"   Continuing without transposition")
                self.apply_transpose = False

    def load_preprocessor_data(self):
        """Load valid placements from preprocessor outputs."""
        logger.info("📂 Loading preprocessor outputs...")

        # Try to load valid placements from parquet first, then csv
        valid_placements_file = None

        parquet_file = self.preprocessor_folder / "valid_placements.parquet"
        csv_file = self.preprocessor_folder / "valid_placements.csv"

        if parquet_file.exists():
            valid_placements_file = parquet_file
            valid_placements = pd.read_parquet(valid_placements_file)
        elif csv_file.exists():
            valid_placements_file = csv_file
            valid_placements = pd.read_csv(valid_placements_file)
        else:
            raise FileNotFoundError(
                f"valid_placements.csv or valid_placements.parquet not found in {self.preprocessor_folder}"
            )

        logger.info(f"   ✓ Loaded valid_placements: {len(valid_placements)} rows")
        logger.info(f"   ✓ Columns: {list(valid_placements.columns)}")

        return valid_placements

    def run_uniform_sampling(self, valid_placements: pd.DataFrame) -> pd.DataFrame:
        """
        Run uniform sampling.

        Parameters:
        -----------
        valid_placements : pd.DataFrame
            Valid placements dataframe from preprocessor

        Returns:
        --------
        pd.DataFrame
            Sampled results
        """
        logger.info("🎲 Running uniform sampling...")
        logger.info(f"   Samples: {self.config['n_events']}")

        # Check if event_id column exists
        if 'event_id' not in valid_placements.columns:
            raise ValueError(
                f"valid_placements must contain 'event_id' column. "
                f"Found columns: {list(valid_placements.columns)}"
            )

        results = sample_storms_uniformly(
            df=valid_placements,
            n_samples=self.config['n_events'],
            seed=self.random_seed
        )

        logger.info(f"   ✓ Sampled {len(results)} events")
        return results

    def apply_storm_transposition(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Apply storm transposition to sampled results.
        Calculates precipitation statistics for transposed storms.

        Parameters:
        -----------
        results : pd.DataFrame
            Sampled results from uniform sampling

        Returns:
        --------
        pd.DataFrame
            Results with transposition data added
        """
        if not self.apply_transpose or self.precip_transpose is None:
            logger.info("⏭️  Skipping transposition")
            return results

        try:
            logger.info("🌊 Applying storm transposition...")

            # Transpose samples to add precipitation
            results = self.precip_transpose.transpose(results)

            logger.info(f"   ✓ Applied transposition")
            logger.info(f"   ✓ Non-null precipitation values: {results['precip_mm'].notna().sum()} / {len(results)}")

            return results

        except Exception as e:
            logger.error(f"❌ Error during transposition: {e}")
            logger.exception("Full traceback:")
            logger.warning("   Continuing without transposition data")
            return results

    def save_results(self, results: pd.DataFrame, export_format: str) -> Path:
        """
        Save sampling results.

        Parameters:
        -----------
        results : pd.DataFrame
            Sampling results
        export_format : str
            Export format ('csv' or 'parquet')

        Returns:
        --------
        Path
            Path to saved file
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format.lower() == 'parquet':
            filename = f"samples_uniform_{timestamp}.parquet"
            filepath = self.output_folder / filename
            results.to_parquet(filepath, index=False, compression='snappy')
            logger.info(f"   ✓ Saved results to {filename}")
        else:  # csv
            filename = f"samples_uniform_{timestamp}.csv"
            filepath = self.output_folder / filename
            results.to_csv(filepath, index=False)
            logger.info(f"   ✓ Saved results to {filename}")

        # Also save summary statistics
        summary = {
            "method": "uniform",
            "n_events": len(results),
            "timestamp": timestamp,
            "columns": list(results.columns),
            "n_rows": len(results),
        }

        # Add transposition summary if available
        if 'precip_mm' in results.columns:
            valid_precip = results['precip_mm'].dropna()
            if len(valid_precip) > 0:
                summary.update({
                    "precip_mm_mean": float(valid_precip.mean()),
                    "precip_mm_std": float(valid_precip.std()),
                    "precip_mm_min": float(valid_precip.min()),
                    "precip_mm_max": float(valid_precip.max()),
                    "precip_mm_count": len(valid_precip),
                })

        summary_file = self.output_folder / "sampling_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SAMPLING SUMMARY\n")
            f.write("=" * 70 + "\n")
            for key, val in summary.items():
                f.write(f"{key}: {val}\n")

        logger.info(f"   ✓ Saved summary to sampling_summary.txt")

        return filepath

    def run(self) -> Path:
        """
        Execute the complete sampling workflow.

        Returns:
        --------
        Path
            Path to output file
        """
        try:
            logger.info("=" * 70)
            logger.info("🚀 Starting Uniform Sampling with Storm Transposition")
            logger.info("=" * 70)

            # Load preprocessor data
            valid_placements = self.load_preprocessor_data()

            # Run uniform sampling
            results = self.run_uniform_sampling(valid_placements)

            # Apply storm transposition if enabled
            results = self.apply_storm_transposition(results)

            # Save results in the same preprocessing folder
            export_format = self.config.get('export_format', 'csv')
            output_file = self.save_results(results, export_format)

            logger.info("=" * 70)
            logger.info(f"✅ Sampling completed successfully!")
            logger.info(f"📊 Results shape: {results.shape}")
            logger.info(f"📊 Columns: {list(results.columns)}")
            logger.info(f"📁 Output: {output_file}")
            logger.info("=" * 70)

            return output_file

        except Exception as e:
            logger.error(f"❌ Sampling failed: {e}")
            logger.exception("Full traceback:")
            raise