#region Imports

#%%
import numpy as np

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.stats.distributions import RotatedNormal

#endregion -----------------------------------------------------------------------------------------
#region Distribution Helpers

#%% Function to fit rotated normal to shapely polygon
def fit_rotated_normal_to_polygon(polygon, coverage_factor=0.5):
    """
    Fits a RotatedNormal distribution to a shapely Polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fit.
        coverage_factor (float): A scaling factor for the standard deviations.
            A value of 1.0 means the std dev will match the vertex spread.
            A value < 1.0 (e.g., 0.5) will "pull in" the distribution to ensure
            more samples fall inside the polygon boundary. Recommended range: 0.3-0.7.

    Returns:
        RotatedNormal: An instance of the RotatedNormal class fitted to the polygon.
    """
    # 1. Get polygon vertices
    # Note: polygon.exterior.coords includes a closing point, which we slice off
    points = np.array(polygon.exterior.coords[:-1])

    # 2. Calculate the mean (centroid)
    # Using polygon.centroid is more accurate than the mean of vertices for irregular shapes
    centroid = np.array(polygon.centroid.coords[0])

    # 3. Perform PCA on the vertices
    # a. Center the data
    centered_points = points - centroid

    # b. Calculate the covariance matrix
    # Note: ddof=1 for sample covariance, which is standard
    cov = np.cov(centered_points, rowvar=False, ddof=1)

    # c. Get eigenvalues and eigenvectors
    # eigh is for symmetric matrices and returns sorted eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 4. Extract parameters
    # The eigenvalues are sorted smallest to largest. We want the largest first.
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # a. The angle is from the first eigenvector (the principal axis)
    # We use arctan2 to get the angle from the vector components
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.rad2deg(angle_rad)

    # b. The standard deviations are the square root of the eigenvalues
    # We apply the coverage_factor here to control how "tight" the fit is.
    stds = np.sqrt(eigenvalues) * coverage_factor

    # print("--- Fit Results ---")
    # print(f"Centroid (Mean): {centroid}")
    # print(f"Angle (Degrees): {angle_deg:.2f}")
    # print(f"Stds (scaled): {stds}")
    # print("--------------------")

    # 5. Create the RotatedNormal distribution with these parameters
    return RotatedNormal(mean=centroid, stds=stds, angle_degrees=angle_deg)

#endregion -----------------------------------------------------------------------------------------
