#region Libraries

#%%
import geopandas as gpd
import os
import pathlib

#endregion -----------------------------------------------------------------------------------------
#region Convert geojson to shp

#%%
path_output = pathlib.Path(r'D:\FEMA Innovations\SO3.1\GIS\Shapefiles')

#%%
path_main = pathlib.Path(r'D:\FEMA Innovations\SO3.1\Data')

#%%
v_path_geojson = list(path_main.glob('**/*.geojson'))

#%%
for path_geojson in v_path_geojson:
    # path_geojson = v_path_geojson[0]
    path_shp = path_output/f'{path_geojson.stem}.shp'

    gdf = gpd.read_file(path_geojson)
    gdf.to_file(path_shp)

#endregion -----------------------------------------------------------------------------------------
#region Generalized Normal Distribution

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm, norm

# Define parameters
loc = 0  # Mean
scale = 1 # Standard deviation equivalent (related to variance)

# Values for the shape parameter beta
beta_normal = 2.0
beta_plateau_moderate = 5.0
beta_plateau_strong = 10.0
beta_laplace = 1.0 # For comparison

# Generate x values
x = np.linspace(-4, 4, 500)

# Calculate PDFs
pdf_normal = gennorm.pdf(x, beta_normal, loc=loc, scale=scale)
# For comparison, you can also use scipy.stats.norm directly
# pdf_scipy_normal = norm.pdf(x, loc=loc, scale=scale) # Will be identical to gennorm with beta=2

pdf_plateau_moderate = gennorm.pdf(x, beta_plateau_moderate, loc=loc, scale=scale)
pdf_plateau_strong = gennorm.pdf(x, beta_plateau_strong, loc=loc, scale=scale)
pdf_laplace = gennorm.pdf(x, beta_laplace, loc=loc, scale=scale)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_normal, label=f'Generalized Normal (beta={beta_normal}, Normal)')
plt.plot(x, pdf_plateau_moderate, label=f'Generalized Normal (beta={beta_plateau_moderate}, Plateau)')
plt.plot(x, pdf_plateau_strong, label=f'Generalized Normal (beta={beta_plateau_strong}, Stronger Plateau)')
plt.plot(x, pdf_laplace, label=f'Generalized Normal (beta={beta_laplace}, Laplace)')


plt.title('Generalized Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# You can also generate random variates
num_samples = 10000
samples_plateau = gennorm.rvs(beta_plateau_moderate, loc=loc, scale=scale, size=num_samples)

plt.figure(figsize=(10,6))
plt.hist(samples_plateau, bins=50, density=True, alpha=0.7, label=f'Samples (beta={beta_plateau_moderate})')
plt.plot(x, pdf_plateau_moderate, 'r-', lw=2, label=f'PDF (beta={beta_plateau_moderate})')
plt.title(f'Histogram of Samples from Generalized Normal (beta={beta_plateau_moderate})')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

#endregion -----------------------------------------------------------------------------------------
#region Generalized Normal Distribution (Truncated)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm

class TruncatedGeneralizedNormal:
    def __init__(self, beta, loc, scale, lower_bound, upper_bound):
        self.beta = beta
        self.loc = loc
        self.scale = scale
        self.m = lower_bound
        self.n = upper_bound

        # Create the underlying (untruncated) generalized normal distribution
        self.untruncated_dist = gennorm(beta=self.beta, loc=self.loc, scale=self.scale)

        # Precompute CDF values at bounds for normalization and sampling
        self.cdf_m = self.untruncated_dist.cdf(self.m)
        self.cdf_n = self.untruncated_dist.cdf(self.n)

        if self.cdf_m >= self.cdf_n:
            raise ValueError(f"Lower bound CDF {self.cdf_m} must be less than upper bound CDF {self.cdf_n}. "
                             f"Check bounds [{self.m}, {self.n}] relative to distribution loc={self.loc}, scale={self.scale}.")
        
        self.normalization_factor = self.cdf_n - self.cdf_m
        if self.normalization_factor < 1e-9: # Avoid division by zero or tiny numbers
             print(f"Warning: Normalization factor is very small ({self.normalization_factor}). "
                   f"The truncation interval [{self.m}, {self.n}] might be in the extreme tails of the base distribution.")


    def pdf(self, v):
        v = np.asarray(v)
        # PDF is 0 outside the truncation bounds
        is_within_bounds = (v >= self.m) & (v <= self.n)
        
        pdf_values = np.zeros_like(v, dtype=float)
        
        if np.any(is_within_bounds):
            # Calculate PDF for values within bounds
            untruncated_pdf_values = self.untruncated_dist.pdf(v[is_within_bounds])
            if self.normalization_factor > 0:
                 pdf_values[is_within_bounds] = untruncated_pdf_values / self.normalization_factor
            else: # Should not happen if constructor check passes, but good for robustness
                 pdf_values[is_within_bounds] = np.inf # Or handle as an error

        return pdf_values

    def cdf(self, v):
        v = np.asarray(v)
        cdf_values = np.zeros_like(v, dtype=float)

        # For v < m, CDF is 0
        # For v > n, CDF is 1
        past_upper_bound = v > self.n
        cdf_values[past_upper_bound] = 1.0
        
        # For m <= v <= n
        within_bounds = (v >= self.m) & (v <= self.n)
        if np.any(within_bounds):
            untruncated_cdf_at_v = self.untruncated_dist.cdf(v[within_bounds])
            if self.normalization_factor > 0:
                cdf_values[within_bounds] = (untruncated_cdf_at_v - self.cdf_m) / self.normalization_factor
            else:
                # If normalization is zero, and v is at or above m (and m=n), cdf can be 0 or 1
                # This case implies m and n are very close and/or in extreme tails.
                # For simplicity, if m=v=n, it could be 1.
                cdf_values[within_bounds & (v >= self.n)] = 1.0


        # Ensure CDF is monotonically increasing and bounded [0,1] due to potential float issues
        cdf_values = np.maximum(0, np.minimum(1, cdf_values))
        return cdf_values

    def rvs(self, size=1):
        # Generate uniform samples in the range [CDF(m), CDF(n)]
        u_scaled = np.random.uniform(low=self.cdf_m, high=self.cdf_n, size=size)
        
        # Use the PPF (inverse CDF) of the untruncated distribution
        return self.untruncated_dist.ppf(u_scaled)

    def mean(self):
        # Numerical integration for the mean of the truncated distribution
        # E[X] = integral from m to n of (x * pdf_truncated(x)) dx
        from scipy.integrate import quad
        integrand = lambda v: v * self.pdf(v)
        mean_val, _ = quad(integrand, self.m, self.n, limit=100) # Increased limit for potentially tricky integrals
        return mean_val
    
    def std(self):
        # Numerical integration for variance, then sqrt for std
        # Var[X] = E[X^2] - (E[X])^2
        # E[X^2] = integral from m to n of (x^2 * pdf_truncated(x)) dx
        from scipy.integrate import quad
        current_mean = self.mean()
        integrand_sq = lambda v: (v**2) * self.pdf(v)
        mean_sq_val, _ = quad(integrand_sq, self.m, self.n, limit=100)
        variance = mean_sq_val - current_mean**2
        if variance < 0 and abs(variance) < 1e-9: # Handle tiny negative due to numerical error
            variance = 0
        if variance < 0:
            print(f"Warning: Calculated negative variance ({variance}). Returning NaN for std.")
            return np.nan
        return np.sqrt(variance)

# --- Example Usage ---
# Parameters:
x_mean = 5.0  # Desired mean of the untruncated plateau distribution
a = 2.0       # Half-width of the desired plateau (scale parameter)
beta_plateau = 10.0 # Shape parameter for gennorm (large for plateau)

# Truncation bounds:
m_trunc = 0.0
n_trunc = 10.0

# Create the truncated distribution object
try:
    tg_dist = TruncatedGeneralizedNormal(
        beta=beta_plateau,
        loc=x_mean,
        scale=a,
        lower_bound=m_trunc,
        upper_bound=n_trunc
    )

    # Generate x values for plotting
    x_vals = np.linspace(m_trunc - 1, n_trunc + 1, 500) # Extend slightly beyond bounds for plotting
    x_plot = np.linspace(m_trunc, n_trunc, 400) # For inside bounds

    # Get PDF and CDF
    pdf_vals = tg_dist.pdf(x_vals)
    cdf_vals = tg_dist.cdf(x_vals)

    # Generate random samples
    num_samples = 10000
    samples = tg_dist.rvs(size=num_samples)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot PDF
    axs[0].plot(x_vals, pdf_vals, 'r-', lw=2, label='Truncated Gennorm PDF')
    axs[0].axvline(m_trunc, color='gray', linestyle='--', label=f'Lower bound m={m_trunc}')
    axs[0].axvline(n_trunc, color='gray', linestyle='--', label=f'Upper bound n={n_trunc}')
    axs[0].axvline(x_mean - a, color='blue', linestyle=':', label=f'x_mean-a ({x_mean-a:.1f})')
    axs[0].axvline(x_mean + a, color='blue', linestyle=':', label=f'x_mean+a ({x_mean+a:.1f})')
    axs[0].axvline(x_mean, color='green', linestyle='-.', label=f'x_mean ({x_mean:.1f}) (loc of base)')
    axs[0].set_title(f'Truncated Generalized Normal (beta={beta_plateau}, loc={x_mean}, scale={a})')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Probability Density')
    axs[0].legend()
    axs[0].grid(True)

    # Plot CDF
    axs[1].plot(x_vals, cdf_vals, 'b-', lw=2, label='Truncated Gennorm CDF')
    axs[1].axvline(m_trunc, color='gray', linestyle='--')
    axs[1].axvline(n_trunc, color='gray', linestyle='--')
    axs[1].set_title('Cumulative Distribution Function (CDF)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Histogram of samples
    axs[2].hist(samples, bins=50, density=True, alpha=0.7, label=f'Histogram of {num_samples} samples')
    axs[2].plot(x_plot, tg_dist.pdf(x_plot), 'r-', lw=2, label='Theoretical PDF') # Plot PDF over histogram
    axs[2].set_title('Histogram of Random Samples')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Density')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print mean and std of the truncated distribution
    print(f"Untruncated distribution loc (target mean for plateau center): {x_mean}")
    print(f"Untruncated distribution scale (target plateau half-width 'a'): {a}")
    print(f"Truncation bounds: [{m_trunc}, {n_trunc}]")
    print(f"Mean of the truncated distribution: {tg_dist.mean():.4f}")
    print(f"Std Dev of the truncated distribution: {tg_dist.std():.4f}")
    print(f"Fraction of original probability mass within bounds: {tg_dist.normalization_factor:.4f}")

except ValueError as e:
    print(f"Error creating truncated distribution: {e}")

#endregion -----------------------------------------------------------------------------------------
#region Tests

# #%%
# #%%
# (pn.ggplot(df_depths_is_1, mapping=pn.aes(x='depth', y='prob'))
#     + pn.geom_point(size=0.1)
# )

# #%%
# (pn.ggplot(df_freq_curve_is_1, mapping=pn.aes(x='depth', y='prob_exceed'))
#     + pn.geom_point(size=0.1)
# )

# #%%
# # df_prob = df_storm_sample_mc_0.copy()    
# df_prob = df_storm_sample_is_1.copy()    
# # df_prob = df_freq_curve_mc_1.copy()    
# # df_prob = df_freq_curve_is_1.copy()    
# df_prob = \
# (df_prob
#     .assign(depth_bin = lambda _: pd.cut(_.y_sampled, bins = 100))
#     .groupby('depth_bin')
#     .agg(prob_count = ('prob', 'size'),
#          prob_mean = ('prob', 'mean'),
#          prob_sum = ('prob', 'sum'))
#     .reset_index()
#     .assign(depth = lambda _: (_.depth_bin.apply(lambda _x: _x.left).astype(float)+_.depth_bin.apply(lambda _x: _x.right).astype(float))/2)
# )
# (pn.ggplot(mapping=pn.aes(x='depth'))
#     # + pn.geom_point(data=df_prob_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc_0/1000}k)"'), size=0.1)
#     # + pn.geom_point(data=df_prob_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_mc_1/1000}k)"'), size=0.1)
#     # + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_count', color=f'"IS ({n_sim_is_1/1000}k), count"'), size=0.1)
#     + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_sum', color=f'"IS ({n_sim_is_1/1000}k), sum"'), size=0.1)
#     # + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_mean', color=f'"IS ({n_sim_is_1/1000}k), mean"'), size=0.1)
#     # + pn.scale_x_log10()
#     + pn.labs(
#         x = 'value',
#         y = 'Probability',
#         title = 'Basic Monte Carlo vs Importance Sampling'
#     )
#     + pn.theme_bw()
#     + pn.theme(
#         title = pn.element_text(hjust = 0.5),
#         # legend_position = 'bottom',
#         legend_title = pn.element_blank(),
#         legend_key = pn.element_blank(),
#         axis_title_y = pn.element_text(ha = 'left'),
#     )
# )

#endregion -----------------------------------------------------------------------------------------
