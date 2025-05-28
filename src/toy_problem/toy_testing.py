#region Libraries

import os
import random
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, uniform, multivariate_normal
import plotnine as pn
from tqdm import tqdm
from typing import Literal

#endregion -----------------------------------------------------------------------------------------

#region Model

class WatershedModel:
    """Model for generating and managing watershed cells."""

    @staticmethod
    def generate_rectangular_cells(transposition_domain_size, size_x, size_y):
        """
        Generate a rectangular watershed within the given domain size.

        Args:
            transposition_domain_size (tuple): Size of the domain (width, height).
            size_x (int): Width of the rectangular watershed.
            size_y (int): Height of the rectangular watershed.

        Returns:
            set: A set of tuples representing the rectangular watershed cells.
        """
        try:
            transposition_domain_size_x, transposition_domain_size_y = transposition_domain_size
            start_x = random.randint(0, transposition_domain_size_x - size_x)
            start_y = random.randint(0, transposition_domain_size_y - size_y)
            return {(start_x + i, start_y + j) for i in range(size_x) for j in range(size_y)}
        except Exception as e:
            print(f"Error in generate_rectangular_cells: {e}")
            raise

    @staticmethod
    def generate_random_cells(transposition_domain_size, area, aspect=1):
        """
        Generate a randomly shaped and contiguous watershed of a specified area.

        Args:
            transposition_domain_size (tuple): Size of the domain (width, height).
            area (int): Total number of cells in the watershed.
            aspect (float): Aspect ratio for the watershed shape.

        Returns:
            set: A set of tuples representing the randomly shaped watershed cells.
        """
        try:
            transposition_domain_size_x, transposition_domain_size_y = transposition_domain_size

            def is_contiguous(cells):
                """
                Check if all cells in the watershed are contiguous.

                Args:
                    cells (set): Set of tuples representing watershed cells.

                Returns:
                    bool: True if all cells are contiguous, False otherwise.
                """
                visited = set()
                stack = [next(iter(cells))]
                while stack:
                    cell = stack.pop()
                    if cell not in visited:
                        visited.add(cell)
                        x, y = cell
                        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                        stack.extend([neighbor for neighbor in neighbors if neighbor in cells])
                return visited == cells

            while True:
                watershed_cells = set()
                start_x = random.randint(1, transposition_domain_size_x)
                start_y = random.randint(1, transposition_domain_size_y)
                watershed_cells.add((start_x, start_y))
                while len(watershed_cells) < area:
                    current_cell = random.choice(list(watershed_cells))
                    x, y = current_cell
                    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                    valid_neighbors = [cell for cell in neighbors if 1 <= cell[0] <= transposition_domain_size_x and 1 <= cell[1] <= transposition_domain_size_y]
                    if valid_neighbors:
                        new_cell = random.choice(valid_neighbors)
                        watershed_cells.add(new_cell)
                if is_contiguous(watershed_cells):
                    return watershed_cells
        except Exception as e:
            print(f"Error in generate_random_cells: {e}")
            raise

    @staticmethod
    def calculate_stats(cells):
        """
        Calculate statistics for a given set of watershed cells.

        Args:
            cells (set): Set of tuples representing watershed cells.

        Returns:
            dict: A dictionary containing centroid, min, max, and range of the cells.
        """
        try:
            x_coords = [cell[0] for cell in cells]
            y_coords = [cell[1] for cell in cells]
            return {
                "centroid": (np.mean(x_coords), np.mean(y_coords)),
                "min": (np.min(x_coords), np.min(y_coords)),
                "max": (np.max(x_coords), np.max(y_coords)),
                "range": (np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords))
            }
        except Exception as e:
            print(f"Error in calculate_stats: {e}")
            raise


class StormModel:
    """Model for generating storm templates and properties."""

    @staticmethod
    def generate_templates(count, min_size, max_size, mean=10, std=3, noise_level=0.1):
        """
        Generate a specified number of storm templates with random sizes, values, and intensities.

        Args:
            count (int): Number of storm templates to generate.
            min_size (int): Minimum size of the storm.
            max_size (int): Maximum size of the storm.
            mean (float): Mean value for storm intensity.
            std (float): Standard deviation for storm intensity.
            noise_level (float): Noise level to add to the storm intensity.

        Returns:
            dict: A dictionary of storm templates with their properties.
        """
        try:
            names = [f'S_{i}' for i in range(1, count + 1)]
            size_x = np.random.randint(min_size, max_size, count)
            size_y = np.random.randint(min_size, max_size, count)
            values = np.random.normal(mean, std, count)
            intensities = [
                StormModel.generate_spatial_intensity((sx, sy), noise_level) * v
                for sx, sy, v in zip(size_x, size_y, values)
            ]
            return {n: (sx, sy, v, i) for n, sx, sy, v, i in zip(names, size_x, size_y, values, intensities)}
        except Exception as e:
            print(f"Error in generate_templates: {e}")
            raise

    @staticmethod
    def generate_spatial_intensity(template_size, noise_level=0.1):
        """
        Create a spatial intensity map for a storm using a multivariate normal distribution.

        Args:
            template_size (tuple): Size of the storm template (width, height).
            noise_level (float): Noise level to add to the intensity map.

        Returns:
            np.ndarray: A 2D array representing the storm intensity map.
        """
        try:
            x = np.linspace(-template_size[0] / 2, template_size[0] / 2, template_size[0])
            y = np.linspace(-template_size[1] / 2, template_size[1] / 2, template_size[1])
            x, y = np.meshgrid(x, y)
            mean = [0, 0]
            cov = [[template_size[0] / 2, 0], [0, template_size[1] / 2]]
            intensity = multivariate_normal(mean, cov).pdf(np.dstack((x, y)))
            intensity /= intensity.max()
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, intensity.shape)
                intensity = np.clip(intensity + noise, 0, 1)
            return intensity
        except Exception as e:
            print(f"Error in generate_spatial_intensity: {e}")
            raise

#endregion -----------------------------------------------------------------------------------------

#region View

class PlotView:
    """View for visualizing data."""

    @staticmethod
    def plot_domain(transposition_domain_size, watershed_cells):
        """
        Create a plot of the transposition domain, highlighting the watershed cells.

        Args:
            transposition_domain_size (tuple): Size of the domain (width, height).
            watershed_cells (set): Set of tuples representing watershed cells.

        Returns:
            plotnine.ggplot: A plot of the domain and watershed.
        """
        try:
            data = [
                {"x": x, "y": y, "fill": "Watershed" if (x, y) in watershed_cells else "Domain"}
                for x in range(transposition_domain_size[0])
                for y in range(transposition_domain_size[1])
            ]
            df_grid = pd.DataFrame(data)
            return (
                pn.ggplot(df_grid, pn.aes(x="x", y="y", fill="fill"))
                + pn.geom_tile(color="black")
                + pn.scale_fill_manual(values={"Domain": "white", "Watershed": "green"})
                + pn.labs(title="Transposition Domain with Watershed Cells", x="X", y="Y")
            )
        except Exception as e:
            print(f"Error in plot_domain: {e}")
            raise

    @staticmethod
    def save_plot(plot, filename):
        """
        Save the plot to the specified folder.

        Args:
            plot (plotnine.ggplot): The plot to save.
            filename (str): The name of the file to save the plot as.
        """
        try:
            output_dir = "data/1_interim/plots"
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            filepath = os.path.join(output_dir, filename)
            plot.save(filepath, width=10, height=8, dpi=300)
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Error in save_plot: {e}")
            raise

#endregion -----------------------------------------------------------------------------------------

