#region Libraries

#%%
import os
import glob
import rasterio
import numpy as np
import pandas as pd

from src.utils_spatial.raster_stats import get_raster_stats

from tqdm import tqdm

#endregion -----------------------------------------------------------------------------------------

#%%
def get_raster_stats(path_raster, sp_domain = None):
    d_stats = {
        'filename': os.path.basename(path_raster),
        'max': np.nan,
        'mean': np.nan,
        'median': np.nan,
        'std': np.nan,
        'cv': np.nan,
        'non_zero_prop': np.nan,
        'mean_non_zero': np.nan,
        'median_non_zero': np.nan,
    }

    try:
        with rasterio.open(path_raster) as src:
            # Read the first band
            if sp_domain is None:
                data = src.read(1).astype(np.float64) # Ensure float for calculations
            else:
                data = rasterio.mask.mask(src, sp_domain.geometry, crop=True)
            nodata_val = src.nodata

            # # Get raster dimensions
            # rows, cols = src.height, src.width

        # Handle NoData values by converting them to NaN
        if nodata_val is not None:
            data[data == nodata_val] = np.nan

        # Flatten row x columns to vector
        data_flat = data_flat.flatten()

        # Remove NaN values for basic stats (std, cv)
        data_flat_non_na = data_flat[~np.isnan(data_flat)]

        # Non-zero values
        data_flat_non_zero = data_flat_non_na[~(data_flat_non_na == 0)]

        if data_flat_non_na.size < 2: # Need at least 2 valid points for std
            print(f"  Not enough valid data in {os.path.basename(path_raster)} for basic stats. Skipping.")
            return d_stats

        # Stats
        max_val = np.max(data_flat)

        mean_val = np.nanmean(data_flat) # nanmean ignores NaNs
        median_val = np.median(data_flat)
        std_val = np.nanstd(data_flat) # nanstd ignores NaNs
        cv = (std_val/mean_val)*100 if mean_val != 0 and not np.isnan(mean_val) else np.nan # Avoid division by zero or if mean is NaN

        zero_prop = 1 - data_flat_non_zero.shape[0] / data_flat_non_na.shape[0]
        mean_val_non_zero = np.nanmean(data_flat_non_zero) # nanmean ignores NaNs
        median_val_non_zero = np.median(data_flat_non_zero)

        d_stats['max'] = max_val
        d_stats['mean'] = mean_val
        d_stats['median'] = median_val
        d_stats['std'] = std_val
        d_stats['cv'] = cv
        d_stats['zero_prop'] = zero_prop
        d_stats['mean_non_zero'] = mean_val_non_zero
        d_stats['median_non_zero'] = median_val_non_zero

        return d_stats

    except Exception as e:
        print(f"Error processing {os.path.basename(path_raster)}: {e}")
        return d_stats

#%%
def main():
    # --- User Configuration ---
    input_folder = r"W:\Water3\Projects\206698_FEMA_SO3_Inno\Calcs\Working\PB\_Scripts\20250602\Importance-Sampling-for-SST\data\1_interim\Trinity\data\storm_catalog\tifs"  # <<< CHANGE THIS to your folder with TIF files
    output_csv = r"W:\Water3\Projects\206698_FEMA_SO3_Inno\Calcs\Working\PB\_Scripts\20250602\Importance-Sampling-for-SST\data\1_interim\Trinity\others\storms.csv" # <<< Name of the output CSV file
    # --- End User Configuration ---

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    if not tif_files:
        tif_files = glob.glob(os.path.join(input_folder, "*.tiff")) # Also check for .tiff
    
    if not tif_files:
        print(f"No .tif or .tiff files found in '{input_folder}'.")
        return

    print(f"Found {len(tif_files)} TIF files to process.")
    pbar = tqdm(total=len(tif_files))
    results = []
    for tif_file in tif_files:
        # print(f"Processing {os.path.basename(tif_file)}...")
        analysis_result = get_raster_stats(tif_file)
        results.append(analysis_result)
        pbar.update()

    # Convert results to a Pandas DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    # print(f"\nAnalysis complete. Results saved to '{output_csv}'")

if __name__ == "__main__":
    main()