#region Libraries

#%%
# import os

import random

import pandas as pd
import numpy as np

from scipy.stats import truncnorm, uniform, multivariate_normal

# import matplotlib.pyplot as plt
# import plotly.express as px
import plotnine as pn

from tqdm import tqdm

from typing import Literal

#endregion -----------------------------------------------------------------------------------------
#region Working Folder

#%%
# os.chdir(r'D:\FEMA Innovations\SO3.1\Py')

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Function to generate randomly positioned rectangular watershed cells
def generate_watershed_cells_rect(transposition_domain_size, size_x, size_y):
    # Transposition domain size
    transposition_domain_size_x = transposition_domain_size[0]
    transposition_domain_size_y = transposition_domain_size[1]

    # Randomly select a starting cell within the domain ensuring the watershed fits within the domain
    start_x = random.randint(0, transposition_domain_size_x - size_x)
    start_y = random.randint(0, transposition_domain_size_y - size_y)

    # Generate watershed cells
    watershed_cells = {(start_x + i, start_y + j) for i in range(size_x) for j in range(size_y)}

    return watershed_cells

#%% Function to generate randomly shaped and positioned contiguous watershed cells
def generate_watershed_cells_rand(transposition_domain_size, area, aspect=1):
    # Transposition domain size
    transposition_domain_size_x = transposition_domain_size[0]
    transposition_domain_size_y = transposition_domain_size[1]

    # Function to check if all cells are contiguous
    def is_contiguous(cells):
        visited = set()
        stack = [next(iter(cells))]
        
        while stack:
            cell = stack.pop()
            if cell not in visited:
                visited.add(cell)
                x, y = cell
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                valid_neighbors = [neighbor for neighbor in neighbors if neighbor in cells]
                stack.extend(valid_neighbors)
        
        return visited == cells

    while True:
        # Initialize watershed cells list
        watershed_cells = set()
        
        # Randomly select a starting cell within the domain
        start_x = random.randint(1, transposition_domain_size_x)
        start_y = random.randint(1, transposition_domain_size_y)
        watershed_cells.add((start_x, start_y))
        
        # Generate contiguous cells until the desired area is reached
        while len(watershed_cells) < area:
            # Randomly select a cell from the current watershed cells
            current_cell = random.choice(list(watershed_cells))
            x, y = current_cell
            
            # Generate possible neighboring cells
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            valid_neighbors = [cell for cell in neighbors if 1 <= cell[0] <= transposition_domain_size_x and 1 <= cell[1] <= transposition_domain_size_y]

            if aspect != 1:
                p = [1, 1, aspect, aspect]
                p = np.array([p for (p, cell) in zip(p, neighbors) if 1 <= cell[0] <= transposition_domain_size_x and 1 <= cell[1] <= transposition_domain_size_y])
                p = p/p.sum()

            # Randomly select a valid neighboring cell and add it to the watershed cells
            if valid_neighbors:
                if aspect == 1:
                    new_cell = random.choice(valid_neighbors)
                else:
                    new_cell_index = np.random.choice(np.arange(len(valid_neighbors)), p=p)
                    new_cell = valid_neighbors[new_cell_index]

                watershed_cells.add(new_cell)
        
        # Check if all cells are contiguous
        if is_contiguous(watershed_cells):
            return watershed_cells
        
#%% Function to generate spatially varying storm intensity
def generate_storm_spatial(template_size, noise_level=0.1):
    # Create a grid for the storm
    x = np.linspace(-template_size[0] / 2, template_size[0] / 2, template_size[0] * 1)
    y = np.linspace(-template_size[1] / 2, template_size[1] / 2, template_size[1] * 1)
    x, y = np.meshgrid(x, y)
    
    # Define the mean (center) of the storm
    mean = [0, 0]
    
    # Define the covariance matrix to control the spread of the storm
    cov = [[template_size[0] / 2, 0], [0, template_size[1] / 2]]
    
    # Generate the storm intensity using a multivariate normal distribution
    storm_intensity = multivariate_normal(mean, cov).pdf(np.dstack((x, y)))
    
    # Normalize the storm intensity to have values between 0 and 1
    storm_intensity /= storm_intensity.max()
    
    if noise_level == 0:
        return storm_intensity
    else:
        # Add noise to the storm intensity
        noise = np.random.normal(0, noise_level, storm_intensity.shape)
        storm_intensity_with_noise = storm_intensity + noise
        
        # Ensure the values are within the range [0, 1]
        storm_intensity_with_noise = np.clip(storm_intensity_with_noise, 0, 1)
    
        return storm_intensity_with_noise

#%% Function to generate random storm sizes, values, and intensities
def generate_storm_templates(count, min_size, max_size, mean = 10, std = 3, noise_level=0.1):
    name = [f'S_{i}' for i in range(1, count+1)]

    size_x = np.random.randint(min_size, max_size, count)
    size_y = np.random.randint(min_size, max_size, count)
    value = np.random.normal(mean, std, count)

    intensity = [generate_storm_spatial((ts_x, ts_y), noise_level)*v for ts_x, ts_y, v in zip(size_x, size_y, value)]

    # return set(zip(size_x, size_y, value))
    d = {n: (sx, sy, v, i) for n, sx, sy, v, i in zip(name, size_x, size_y, value, intensity)}

    return d

#%% Plot domain and watershed
def plot_domain(transposition_domain_size, watershed_cells):
    # Transposition domain size
    transposition_domain_size_x = transposition_domain_size[0]
    transposition_domain_size_y = transposition_domain_size[1]

    # Prepare data for plotnine
    data = []
    for x in range(transposition_domain_size_x):
        for y in range(transposition_domain_size_y):
            if (x, y) in watershed_cells:
                data.append({'x': x, 'y': y, 'fill': 'Watershed'})
            else:
                data.append({'x': x, 'y': y, 'fill': 'Domain'})
    df_grid = pd.DataFrame(data)

    # Create plot using plotnine
    g = \
    (pn.ggplot(df_grid, pn.aes(x='x', y='y', fill='fill'))
        + pn.geom_tile(color='black')
        + pn.scale_fill_manual(values={'Domain': 'white', 'Watershed': 'green'})
        # + pn.theme_minimal()
        + pn.labs(
            title='Transposition Domain with Watershed Cells', 
            x='X', 
            y='Y'
        )
    )

    return (g)

#%% Function to check if storm overlaps with watershed
def storm_intersection_check(watershed_cells, storm_x_range, storm_y_range):
    storm_cells = {(x, y) for x in storm_x_range for y in storm_y_range}
    return not watershed_cells.isdisjoint(storm_cells)

#%% Find intersection area between watershed and storm
def storm_intersection_area(watershed_cells, storm_x_range, storm_y_range):
    storm_cells = {(x, y) for x in storm_x_range for y in storm_y_range}
    return watershed_cells.intersection(storm_cells)

#%% Find storm cell intensity values that intersect watershed
def storm_intersection_values(watershed_cells, storm_x_range, storm_y_range, storm_intensities):
    intersected_intensities = []
    
    for cell in watershed_cells:
        x, y = cell
        if x in storm_x_range and y in storm_y_range:
            storm_x_index = storm_x_range.index(x)
            storm_y_index = storm_y_range.index(y)
            intersected_intensity = storm_intensities[storm_y_index, storm_x_index]
            intersected_intensities.append(intersected_intensity)
    
    return np.array(intersected_intensities)

#%% Function to pass parameters to truncnorm
def truncnorm_params(mean, std_dev, lower, upper):
    d = dict(
        a = (lower - mean) / std_dev, 
        b = (upper - mean) / std_dev, 
        loc = mean, 
        scale = std_dev
    )

    return d    

#%% Function to calculate centroid, min, max, range from cells
def calculate_cell_stats(cells):
    x_coords = [cell[0] for cell in cells]
    y_coords = [cell[1] for cell in cells]

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    min_x = np.min(x_coords)
    min_y = np.min(y_coords)

    max_x = np.max(x_coords)
    max_y = np.max(y_coords)

    range_x = max_x - min_x
    range_y = max_y - min_y

    d = dict(
        centroid = (centroid_x, centroid_y),
        min = (min_x, min_y),
        max = (max_x, max_y),
        range = (range_x, range_y)
    )

    return d

#%% Create probability dataframe from depths (sorted) and weights (sorted)
def get_df_prob(depths, weights):
    # Table of depths and probabilities
    df_prob_mc = pd.DataFrame(dict(
        depth = depths,
        prob = weights
    ))
    
    # Exceedence probability
    df_prob_mc = \
    (df_prob_mc
        .assign(prob_exceed = lambda _: _.prob.cumsum())
        .assign(return_period = lambda _: 1/_.prob_exceed)
    )

    return df_prob_mc    

#%% Get storm position in domain
def get_storm_properties(storm_templates, centroid_x, centroid_y):
    # Sample storm from storm_templates
    storm_shape_key = np.random.choice(list(storm_templates.keys()))
    storm_shape = storm_templates[storm_shape_key]
    
    # Calculate half width for the current sampled storm
    half_width_x = int(storm_shape[0] / 2)
    half_width_y = int(storm_shape[1] / 2)
    
    # Determine storm ranges (can be outside the domain)
    start_x = centroid_x - half_width_x
    end_x = start_x + storm_shape[0]
    
    start_y = centroid_y - half_width_y
    end_y = start_y + storm_shape[1]

    storm_x_range = range(start_x, end_x)
    storm_y_range = range(start_y, end_y)

    # Storm values
    storm_value = storm_shape[2]
    storm_intensities = storm_shape[3]

    return (storm_x_range, storm_y_range, storm_value, storm_intensities)

#%% Get storm depth based on intersection between watershed and storm
def get_depth(watershed_cells, storm_properties, method_value: Literal['normal', 'storm', 'overlap', 'overlap_normal', 'total_intensity_values'] = 'total_intensity_values'):
    # Get storm properties (range, value, and intensities) 
    storm_x_range, storm_y_range, storm_value, storm_intensities = storm_properties

    # Intersection between watershed and storm
    intersection_cells = storm_intersection_area(watershed_cells, storm_x_range, storm_y_range)
    intersection_area = len(intersection_cells)
    
    # Check if the storm overlaps with the watershed
    if intersection_area > 0:
        if method_value == 'normal':
            # Sample storm depth from normal distribution N(10, 2)
            depth = np.random.normal(10, 2)
        elif method_value == 'storm':
            depth = storm_value
        elif method_value == 'overlap':
            depth = storm_value*intersection_area**0.1
        elif method_value == 'overlap_normal':
            depth = np.random.normal(storm_value, 1)*intersection_area**0.1
        elif method_value == 'total_intensity_values':
            depth = storm_intersection_values(watershed_cells, storm_x_range, storm_y_range, storm_intensities).sum()
        
        if depth < 0:
            depth = 0
    else:
        depth = 0

    return depth      

#%% Basic Monte Carlo Sampling
def get_freq_df_monte_carlo(transposition_domain_size, storm_templates, watershed_cells, num_simulations, method_value: Literal['normal', 'storm', 'overlap', 'overlap_normal', 'total_intensity_values'] = 'total_intensity_values'):
    # Transposition domain size
    transposition_domain_size_x = transposition_domain_size[0]
    transposition_domain_size_y = transposition_domain_size[1]
    
    # Iteration to get storm depths
    v_centroid_x = np.random.uniform(1, transposition_domain_size_x + 1, size = num_simulations).round().astype(int)
    v_centroid_y = np.random.uniform(1, transposition_domain_size_y + 1, size = num_simulations).round().astype(int)
    v_depth = []
    pbar = tqdm(total=num_simulations)
    for i in range(num_simulations):
        # Sample centroid of storm from uniform distributions
        centroid_x = v_centroid_x[i]
        centroid_y = v_centroid_y[i]
        
        # Get storm properties
        storm_properties = get_storm_properties(storm_templates, centroid_x, centroid_y)
        
        # Get depth
        depth = get_depth(watershed_cells, storm_properties, method_value=method_value)

        # Record values
        v_depth.append(depth)
        
        # Update progress bar
        pbar.update(1)

    # Give equal weight to each depth
    v_weight = [1 / (len(v_depth) + 1)] * len(v_depth)

    # Dataframe of centroids, depths, and weights
    df = pd.DataFrame(dict(
        x = v_centroid_x,
        y = v_centroid_y,
        depth = v_depth,
        prob = v_weight,
    ))
    df = df.sort_values('depth', ascending=False)

    # Get exceedence probability and return period
    _df_T = get_df_prob(df.depth.values, df.prob.values)
    df = \
    (df
        .assign(prob_exceed = _df_T.prob_exceed.values)
        .assign(return_period = _df_T.return_period.values)
    )

    return df

#%% Importance Sampling
def get_freq_df_importance_sampling(transposition_domain_size, storm_templates, watershed_cells, dist_x, dist_y, num_simulations, method_value: Literal['normal', 'storm', 'overlap', 'total_intensity_values'] = 'total_intensity_values'):
    # Transposition domain size
    transposition_domain_size_x = transposition_domain_size[0]
    transposition_domain_size_y = transposition_domain_size[1]

    # Iteration to get storm depths and weights
    v_centroid_x = dist_x.rvs(num_simulations).round().astype(int)
    v_centroid_y = dist_y.rvs(num_simulations).round().astype(int)
    v_depth = []
    v_weight = []
    pbar = tqdm(total=num_simulations)
    for i in range(num_simulations):
        # Sample centroid of storm from truncated normal distributions
        centroid_x = v_centroid_x[i]
        centroid_y = v_centroid_y[i]
        
        # Get storm properties
        storm_properties = get_storm_properties(storm_templates, centroid_x, centroid_y)
        
        # Get depth
        depth = get_depth(watershed_cells, storm_properties, method_value=method_value)

        # Compute weight of each depth
        f_X_U = 1 / transposition_domain_size_x
        f_Y_U = 1 / transposition_domain_size_y
        f_X_TN = dist_x.pdf(centroid_x)
        f_Y_TN = dist_y.pdf(centroid_y)
        p = f_X_U * f_Y_U
        q = f_X_TN * f_Y_TN
        weight = p / q if q > 0 else 0

        # Record values
        v_depth.append(depth)
        v_weight.append(weight)

        # Update progress bar
        pbar.update(1)

    # Normalize weights
    v_weight = np.array(v_weight)
    v_weight /= np.sum(v_weight)

    # Dataframe of centroids, depths, and weights
    df = pd.DataFrame(dict(
        x = v_centroid_x,
        y = v_centroid_y,
        depth = v_depth,
        prob = v_weight,
    ))
    df = df.sort_values('depth', ascending=False)

    # Get exceedence probability and return period
    _df_T = get_df_prob(df.depth.values, df.prob.values)
    df = \
    (df
        .assign(prob_exceed = _df_T.prob_exceed.values)
        .assign(return_period = _df_T.return_period.values)
    )

    return df

#%% Print simulation statistics
def print_sim_stats(df_prob):
    n_sim = df_prob.shape[0]
    n_sim_intersect = df_prob.loc[lambda _: _.depth > 0].shape[0]
    rate_success = n_sim_intersect/n_sim*100

    df_prob = \
    (df_prob
        .assign(x_px = lambda _: _.depth * _.prob)
    )
    mean = df_prob.x_px.sum()
    df_prob = \
    (df_prob
        .assign(x_mx_px = lambda _: ((_.depth - mean)**2) * _.prob)
    )
    std = np.sqrt(df_prob.x_mx_px.sum())
    standard_error = std/np.sqrt(n_sim)

    print(
        f'{n_sim_intersect} out of {n_sim} ({rate_success:.2f}%)\n'
        + f'Result: {mean:.2f} ± {standard_error:.2f}'
    )

#endregion -----------------------------------------------------------------------------------------
#region Main

#%% Define constants
# transposition_domain_size = (20, 20)
transposition_domain_size = (100, 100)

#%% Define storm templates
# storm_templates = {
#     'A': (2, 2, 45, []),
#     'B': (4, 4, 80, []),
#     'C': (6, 6, 120, [])
# }
storm_templates = generate_storm_templates(50, 5, 50, mean=30, std=3, noise_level=0.1)

#%% Watershed cells
# watershed_cells = {(x, y) for x in range(3, 7) for y in range(2, 6)}
# watershed_cells = {(x, y) for x in range(3, 12) for y in range(2, 4)}
# watershed_cells = generate_watershed_cells_rect(transposition_domain_size, 15, 15)
# watershed_cells = generate_watershed_cells_rand(transposition_domain_size, 400)
watershed_cells = generate_watershed_cells_rand(transposition_domain_size, 225, aspect=1/20)

plot_domain(transposition_domain_size, watershed_cells)

#%% Distribution for importance sampling
watershed_stats = calculate_cell_stats(watershed_cells)

# dist_x = uniform(1, transposition_domain_size[0])
# dist_y = uniform(1, transposition_domain_size[1])

# dist_x = uniform(watershed_stats.get('min')[0], watershed_stats.get('max')[0])
# dist_y = uniform(watershed_stats.get('min')[1], watershed_stats.get('max')[1])

dist_x = truncnorm(**truncnorm_params(watershed_stats.get('centroid')[0], watershed_stats.get('range')[0]+1, 1, transposition_domain_size[0]))
dist_y = truncnorm(**truncnorm_params(watershed_stats.get('centroid')[1], watershed_stats.get('range')[1]+1, 1, transposition_domain_size[1]))

# dist_x = Dist_Truncated_Normal(watershed_stats.get('centroid')[0], watershed_stats.get('range')[0]+1, 1, transposition_domain_size[0])
# dist_y = Dist_Truncated_Normal(watershed_stats.get('centroid')[1], watershed_stats.get('range')[1]+1, 1, transposition_domain_size[1])

#%% Simulation Parameters
method_value = ['normal', 'storm', 'overlap', 'overlap_normal', 'total_intensity_values'][0]
n_sim_mc_0 = 10000
n_sim_mc_1 = 1000
n_sim_is_1 = 1000

#%% Perform simulation
tqdm._instances.clear()
df_prob_mc_0 = get_freq_df_monte_carlo(transposition_domain_size, storm_templates, watershed_cells, n_sim_mc_0, method_value=method_value)
df_prob_mc_1 = get_freq_df_monte_carlo(transposition_domain_size, storm_templates, watershed_cells, n_sim_mc_1, method_value=method_value)
df_prob_is_1 = get_freq_df_importance_sampling(transposition_domain_size, storm_templates, watershed_cells, dist_x, dist_y, n_sim_is_1, method_value=method_value)

#%% Plot frequency curves
(pn.ggplot(mapping=pn.aes(x='return_period', y='depth'))
    + pn.geom_point(data=df_prob_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc_0/1000}k)"'), size=0.1)
    + pn.geom_point(data=df_prob_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_mc_1/1000}k)"'), size=0.1)
    + pn.geom_point(data=df_prob_is_1, mapping=pn.aes(color=f'"IS ({n_sim_is_1/1000}k)"'), size=0.1)
    + pn.scale_x_log10()
    + pn.labs(
        x = 'Return Period',
        y = 'Rainfall Depth',
        title = 'Basic Monte Carlo vs Importance Sampling'
    )
    + pn.theme_bw()
    + pn.theme(
        title = pn.element_text(hjust = 0.5),
        # legend_position = 'bottom',
        legend_title = pn.element_blank(),
        legend_key = pn.element_blank(),
        axis_title_y = pn.element_text(ha = 'left'),
    )
)

#%% Plot centroids for importance sampling
(pn.ggplot(df_prob_is_1, pn.aes(x='x', y='y'))
    + pn.geom_point(mapping=pn.aes(color='prob'), size=0.1)
    + pn.xlim(1, transposition_domain_size[0])
    + pn.ylim(1, transposition_domain_size[1])
    + pn.scale_color_gradient(low='blue', high='red', trans='log10')
)

#%% Show simulation statistics
print_sim_stats(df_prob_mc_0)
print_sim_stats(df_prob_mc_1)
print_sim_stats(df_prob_is_1)


#endregion -----------------------------------------------------------------------------------------
