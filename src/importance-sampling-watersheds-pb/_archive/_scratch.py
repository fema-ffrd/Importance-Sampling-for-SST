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
#region Tests for Importance Sampling

#%% Imports
import numpy as np
import pandas as pd
from scipy import integrate 
from scipy.stats import uniform, norm, truncnorm
import matplotlib.pyplot as plt
import plotnine as pn

#%% Imports (modules)
from modules.distributions import truncnorm_params, TruncatedGeneralizedNormal

#%% Range for x
x_min = 0
x_max = 20

watershed_x_min = 5
watershed_x_max = 10

#%% Function to calculate y
def calc(x):
    # return np.where((x >= watershed_x_min) & (x <= watershed_x_max), (x**3+4**x), 0)
    return np.where((x >= watershed_x_min) & (x <= watershed_x_max), norm(loc=7.5, scale=0.5).pdf(x), 0)

#%% Scalar version for integration
def calc_scalar(x):
    if 5 <= x <= 10:
        # return x**3 + 4**x
        return norm(loc=7.5, scale=0.5).pdf(x)
    else:
        return 0

#%% Real expected value of y
y_exp_real = integrate.quad(calc_scalar, x_min, x_max)[0]/(x_max - x_min)
print (y_exp_real)

#%% Expected value of y using Monte Carlo simulation
dist_mc = uniform(loc=x_min, scale=x_max - x_min)
x_mc = dist_mc.rvs(100_000)
y_mc = calc(x_mc)
y_exp_mc = np.mean(y_mc)
print (y_exp_mc)

#%% Expected value of y using Importance Sampling simulation (TruncNorm)
param_std = 0.8 # at least 0.8 to be good
# dist_is = truncnorm(loc=7.5, scale=param_std, a=(x_min-7.5)/param_std, b=(x_max-7.5)/param_std)
dist_is = truncnorm(**truncnorm_params(7.5, param_std, x_min, x_max))
x_is = dist_is.rvs(100_000)
y_is = calc(x_is)
p = dist_mc.pdf(x_is) # PDF of original uniform distribution at sampled points
q = dist_is.pdf(x_is) # PDF of proposal truncated normal at sampled points
weights = np.where(q > 1e-9, p/q, 0)
y_exp_is = np.mean(y_is * weights)
# y_exp = np.mean(y)
print (y_exp_is)

#%% Expected value of y using Importance Sampling simulation (TruncGenNorm)
beta = 5 # at least ? to be good
dist_is = TruncatedGeneralizedNormal(
    loc=7.5,
    scale=(watershed_x_max-watershed_x_min),
    lower_bound=x_min,
    upper_bound=x_max,
    beta=beta,
)
x_is = dist_is.rvs(100_000)
y_is = calc(x_is)
p = dist_mc.pdf(x_is) # PDF of original uniform distribution at sampled points
q = dist_is.pdf(x_is) # PDF of proposal truncated normal at sampled points
weights = np.where(q > 1e-9, p/q, 0)
y_exp_is = np.mean(y_is * weights)
# y_exp = np.mean(y)
print (y_exp_is)

#%% Probability of exceedence
y_val = 0.0001 #750_000

prob_exceed_mc = np.mean(y_mc >= y_val)

prob_exceed_is = np.mean((y_is >= y_val) * weights)

print (prob_exceed_mc)
print (prob_exceed_is)

#%%
(pn.ggplot(pd.DataFrame(dict(x = x_mc)), pn.aes(x='x'))
    + pn.geom_histogram()
    + pn.lims(x = (x_min, x_max))
)    + pn.labs(x = 'x (monte carlo sampling)')

#%%
(pn.ggplot(pd.DataFrame(dict(x = x_is)), pn.aes(x='x'))
    + pn.geom_histogram()
    + pn.lims(x = (x_min, x_max))
)    + pn.labs(x = 'x (importance sampling)')

#%%
# (pn.ggplot(pd.DataFrame(dict(x = y_mc)), pn.aes(x='x'))
#     + pn.geom_histogram()
#     + pn.scale_y_log10()
#     + pn.labs(x = 'y (real values)')
# )

#%%
(pn.ggplot(pd.DataFrame(dict(x = y_mc)).loc[lambda _: _.x > 0], pn.aes(x='x'))
    + pn.geom_histogram()
    + pn.scale_y_log10()
    + pn.labs(x = 'y (real values, non-zero)')
)

#%%
df_xy_mc_stats = \
(pd.DataFrame(dict(x = x_mc, y = y_mc))
    .groupby('x')
    .agg(y_min=('y', 'min'), y_max=('y', 'max'), y_mean=('y', 'mean'), y_median=('y', 'median'))
    .reset_index()
)
(pn.ggplot(df_xy_mc_stats, pn.aes(x = 'x'))
    + pn.geom_point(pn.aes(y = 'y_median'))
)

#endregion -----------------------------------------------------------------------------------------
#region Tests of Importance Sampling Archive

#%% Imports
import numpy as np
import pandas as pd # Not used in this specific script, can be removed
from scipy import integrate
from scipy.stats import uniform, truncnorm
import matplotlib.pyplot as plt # Keep for visualization

#%% Range for x
x_min = 0
x_max = 20
num_samples_mc = 100_000
num_samples_is = 10_000 # We might need fewer samples for IS to get a good estimate

#%% Function to calculate y
def calc(x):
    # Ensure x is a numpy array for vectorized operations
    x = np.asarray(x)
    return np.where((x >= 5) & (x <= 10), (x**3 + 4**x), 0)

#%% Real expected value of y
# p(x) is the PDF of Uniform(x_min, x_max), which is 1/(x_max - x_min)
pdf_p_x_constant = 1.0 / (x_max - x_min)
# E[y] = ∫ y(x) * p(x) dx
integrand_real = lambda x: calc(x) * pdf_p_x_constant
y_exp_real = integrate.quad(integrand_real, x_min, x_max)[0]
# Alternatively, as you had (which is also correct):
# y_exp_real_v2 = integrate.quad(calc, x_min, x_max)[0] / (x_max - x_min)
print(f"Real expected value of y: {y_exp_real:.6f}")

#%% Expected value of y using standard Monte Carlo simulation
# Original distribution p(x): Uniform(x_min, x_max)
p_dist = uniform(loc=x_min, scale=x_max - x_min)
x_mc = p_dist.rvs(num_samples_mc)
y_mc = calc(x_mc)
y_exp_mc = np.mean(y_mc)
print(f"MC expected value of y ({num_samples_mc} samples): {y_exp_mc:.6f}")

#%% Importance Sampling
print("\n--- Importance Sampling ---")

# 1. Define the original distribution p(x) (already done as p_dist)

# 2. Define the proposal distribution q(x): Truncated Normal
mu_q = 7.5  # Mean of the underlying normal
sigma_q = 1.5 # Standard deviation of the underlying normal (heuristic choice)
# sigma_q = 2.0 # Another option for sigma

# For truncnorm, 'a' and 'b' are standard deviations from the mean if loc and scale are used for the underlying normal.
# We want to truncate the normal distribution N(mu_q, sigma_q^2) to the interval [x_min, x_max].
a_trunc = (x_min - mu_q) / sigma_q  # Lower bound in std devs from mu_q
b_trunc = (x_max - mu_q) / sigma_q  # Upper bound in std devs from mu_q

q_dist = truncnorm(a=a_trunc, b=b_trunc, loc=mu_q, scale=sigma_q)

# 3. Draw samples from q(x)
x_is = q_dist.rvs(num_samples_is)

# 4. Calculate y for these samples
y_is = calc(x_is)

# 5. Calculate importance weights w(x) = p(x) / q(x)
p_pdf_values = p_dist.pdf(x_is) # PDF of original uniform distribution at sampled points
q_pdf_values = q_dist.pdf(x_is) # PDF of proposal truncated normal at sampled points

# Avoid division by zero if any q_pdf_value is zero (should not happen for samples from q_dist itself)
# but good practice if x_is could come from elsewhere or q_dist had zero density regions.
# In our case, p_pdf_values will be non-zero for x_is in [0,20] and q_pdf_values also.
weights = np.zeros_like(q_pdf_values)
# Only calculate weights where q_pdf_values is not zero.
# And where p_pdf_values is not zero (though for uniform over [0,20] it always is within this range)
valid_indices = (q_pdf_values > 1e-9) # Check for very small q_pdf_values
weights[valid_indices] = p_pdf_values[valid_indices] / q_pdf_values[valid_indices]


# 6. Calculate the Importance Sampling estimate
# E_q[y(X) * w(X)]
y_exp_is = np.mean(y_is * weights)
print(f"IS expected value of y ({num_samples_is} samples, mu_q={mu_q}, sigma_q={sigma_q}): {y_exp_is:.6f}")

#%%
# Diagnostic: Effective Sample Size (ESS)
# A common formula for ESS when weights are not normalized to sum to N
# is (sum(weights))^2 / sum(weights^2)
# Or, if we consider normalized weights w_norm = weights / sum(weights): ESS = 1 / sum(w_norm^2)
if np.sum(weights) > 1e-9: # Avoid division by zero if all weights are tiny
    ess = (np.sum(weights)**2) / np.sum(weights**2)
    print(f"Effective Sample Size (ESS): {ess:.2f} (out of {num_samples_is})")
    # An ESS much smaller than N indicates high variance in weights.
else:
    print("Could not calculate ESS (sum of weights is near zero).")


#%% Visualization (optional, but helpful)
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: PDFs of p(x) and q(x)
x_plot = np.linspace(x_min, x_max, 500)
p_plot_pdf = p_dist.pdf(x_plot)
q_plot_pdf = q_dist.pdf(x_plot)

axes[0].plot(x_plot, p_plot_pdf, label=f'p(x) - Uniform({x_min},{x_max})', color='blue')
axes[0].plot(x_plot, q_plot_pdf, label=f'q(x) - TruncNorm(μ={mu_q},σ={sigma_q}, trunc=[{x_min},{x_max}])', color='green')
axes[0].set_title('Original p(x) and Proposal q(x) PDFs')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Function y(x) and y(x)*p(x)
y_plot = calc(x_plot)
axes[1].plot(x_plot, y_plot, label='y(x)', color='red')
axes[1].plot(x_plot, y_plot * p_plot_pdf, label='y(x) * p(x) (Integrand for E[y])', color='purple', linestyle='--')
axes[1].set_title('Function y(x) and Integrand y(x)p(x)')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Sampled points and their weights for IS
# It's hard to visualize weights directly with y_is, so let's show where samples from q(x) fall
# And maybe the magnitude of weights as a scatter plot if helpful
# Scatter plot of IS samples
# Filter for y_is > 0 to see where the "action" is
active_is_samples = x_is[y_is > 0]
active_weights = weights[y_is > 0]

if len(active_is_samples) > 0:
    sc = axes[2].scatter(active_is_samples, y_is[y_is > 0], c=active_weights, cmap='viridis', s=10, alpha=0.7, label='y_is values (color=weight)')
    plt.colorbar(sc, ax=axes[2], label='Importance Weight')
    axes[2].set_title(f'Importance Samples (x_is from q(x)) where y_is > 0')
else:
    axes[2].set_title(f'Importance Samples (no y_is > 0 found with {num_samples_is} samples)')

# Overlay where MC samples would fall (uniformly) for comparison density
# axes[2].hist(x_mc[y_mc > 0], bins=50, density=True, alpha=0.3, label='MC samples (y>0) distribution', color='grey')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y_is value')
axes[2].legend()
axes[2].grid(True)


plt.tight_layout()
plt.show()

# Further check: where are the largest weights?
if len(weights)>0:
    print("\nWeight statistics:")
    print(f"Min weight: {np.min(weights):.4f}, Max weight: {np.max(weights):.4f}, Mean weight: {np.mean(weights):.4f}, Std weight: {np.std(weights):.4f}")
    # The mean of weights should theoretically be close to 1 if q(x) is a valid PDF and covers p(x) support
    # This is because E_q[p(X)/q(X)] = integral (p(x)/q(x)) q(x) dx = integral p(x) dx = 1
    # However, large variance in weights is a sign of trouble.
    
    # Where are the samples from IS concentrated vs where function is non-zero
    samples_in_5_10_is = np.sum((x_is >= 5) & (x_is <= 10))
    print(f"Proportion of IS samples in [5, 10]: {samples_in_5_10_is / num_samples_is:.2%} (target region)")

    samples_in_5_10_mc = np.sum((x_mc >= 5) & (x_mc <= 10))
    print(f"Proportion of MC samples in [5, 10]: {samples_in_5_10_mc / num_samples_mc:.2%} (target region)")





#%% Imports
import numpy as np
# import pandas as pd # Not used in this specific script, can be removed
from scipy import integrate
from scipy.stats import uniform, truncnorm
import matplotlib.pyplot as plt
import seaborn as sns # For easier KDE plots and styling

#%% Range for x
x_min = 0
x_max = 20
num_samples_mc = 100_000
num_samples_is = 10_000 # IS might need fewer samples

#%% Function to calculate y
def calc(x):
    x = np.asarray(x)
    return np.where((x >= 5) & (x <= 10), (x**3 + 4**x), 0)

#%% Real expected value of y (for reference)
pdf_p_x_constant = 1.0 / (x_max - x_min)
integrand_real = lambda x_val: calc(x_val) * pdf_p_x_constant
y_exp_real = integrate.quad(integrand_real, x_min, x_max)[0]
print(f"Real expected value of y: {y_exp_real:.6f}")

# Probabilities for y=0 and y>0
prob_y_eq_0 = ((5 - x_min) + (x_max - 10)) / (x_max - x_min)
prob_y_gt_0 = (10 - 5) / (x_max - x_min)
print(f"Theoretical P(y=0): {prob_y_eq_0:.4f}")
print(f"Theoretical P(y>0): {prob_y_gt_0:.4f}")

#%% Standard Monte Carlo
print("\n--- Standard Monte Carlo ---")
p_dist = uniform(loc=x_min, scale=x_max - x_min)
x_mc = p_dist.rvs(num_samples_mc)
y_mc = calc(x_mc)
y_exp_mc = np.mean(y_mc)
print(f"MC expected value of y ({num_samples_mc} samples): {y_exp_mc:.6f}")

# Separate y_mc values that are zero and non-zero
y_mc_zero = y_mc[y_mc == 0]
y_mc_nonzero = y_mc[y_mc > 0]
print(f"MC: Proportion of y=0 samples: {len(y_mc_zero)/len(y_mc):.4f}")
print(f"MC: Proportion of y>0 samples: {len(y_mc_nonzero)/len(y_mc):.4f}")


#%% Importance Sampling
print("\n--- Importance Sampling ---")
mu_q = 8.5  # Adjusted mean, since y(x) grows rapidly towards x=10
sigma_q = 1.5 # Std dev
a_trunc = (x_min - mu_q) / sigma_q
b_trunc = (x_max - mu_q) / sigma_q
q_dist = truncnorm(a=a_trunc, b=b_trunc, loc=mu_q, scale=sigma_q)

x_is = q_dist.rvs(num_samples_is)
y_is = calc(x_is)

p_pdf_values = p_dist.pdf(x_is)
q_pdf_values = q_dist.pdf(x_is)
weights = np.zeros_like(q_pdf_values)
valid_indices = (q_pdf_values > 1e-9) # Avoid division by zero
weights[valid_indices] = p_pdf_values[valid_indices] / q_pdf_values[valid_indices]
weights[~valid_indices] = 0 # Assign 0 weight if q_pdf is ~0

y_exp_is = np.mean(y_is * weights)
print(f"IS expected value of y ({num_samples_is} samples, mu_q={mu_q}, sigma_q={sigma_q}): {y_exp_is:.6f}")

# Separate y_is values and their weights
y_is_zero_indices = (y_is == 0)
y_is_nonzero_indices = (y_is > 0)

y_is_zero = y_is[y_is_zero_indices]
weights_zero = weights[y_is_zero_indices]

y_is_nonzero = y_is[y_is_nonzero_indices]
weights_nonzero = weights[y_is_nonzero_indices]

# The sum of weights for y_is_nonzero should approximate P(y>0)
# The sum of weights for y_is_zero should approximate P(y=0)
# (if weights are normalized to sum to 1 across all samples, otherwise sum(weights)/N approximates it)
# Or, more directly: sum(weights[y_is>0]) / sum(weights) approximates P(y>0 | sampled by IS)
# which should in turn approximate the true P(y>0).
if np.sum(weights) > 1e-9:
    prop_y_gt_0_is = np.sum(weights_nonzero) / np.sum(weights)
    prop_y_eq_0_is = np.sum(weights_zero) / np.sum(weights)
    print(f"IS: Weighted proportion of y=0 samples: {prop_y_eq_0_is:.4f}")
    print(f"IS: Weighted proportion of y>0 samples: {prop_y_gt_0_is:.4f}")
else:
    print("IS: Sum of weights is too small to estimate proportions.")


#%% Plotting the probability distributions

# Because y can range from 0 to >1,000,000, we'll focus on y > 0 and use a log scale for y-values.
# We'll normalize the histograms for y > 0 to represent conditional probability density p(y | y > 0).

plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False) # sharex=False as scales differ

# --- Plot for Standard Monte Carlo ---
if len(y_mc_nonzero) > 0:
    # Bins for the log-transformed non-zero y values
    # Calculate min and max for binning, careful with log(0) if any y_mc_nonzero are extremely small
    log_y_mc_nonzero = np.log10(y_mc_nonzero[y_mc_nonzero > 1e-9]) # Avoid log(0) or tiny numbers if any
    if len(log_y_mc_nonzero) > 0:
        min_log_y = np.min(log_y_mc_nonzero)
        max_log_y = np.max(log_y_mc_nonzero)
        # Create bins in log space, then convert back for hist input if needed, or plot on log scale directly
        bins_log_mc = np.logspace(min_log_y, max_log_y, 50)

        # Plot histogram on a log x-scale
        axes[0].hist(y_mc_nonzero, bins=bins_log_mc, density=True, alpha=0.7, label=f'MC P(y|y>0) ({len(y_mc_nonzero)} samples)')
        # The density=True here normalizes so the area of this part is 1.
        # To reflect actual probability, this density should be scaled by P(y>0).
        # For visualization of shape, P(y|y>0) is fine.
        axes[0].set_xscale('log')
        axes[0].set_yscale('log') # Also log y-axis for histogram counts/density for better visibility
        axes[0].set_title(f'Distribution of y > 0 (Standard MC)\nNote: P(y=0) ≈ {len(y_mc_zero)/len(y_mc):.2f}')
        axes[0].set_xlabel('y values (log scale)')
        axes[0].set_ylabel('Density (log scale)')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No non-zero y_mc samples to plot.', ha='center', va='center')
else:
    axes[0].text(0.5, 0.5, 'No non-zero y_mc samples.', ha='center', va='center')
axes[0].grid(True, which="both", ls="-", alpha=0.5)


# --- Plot for Importance Sampling ---
if len(y_is_nonzero) > 0 and np.sum(weights_nonzero) > 1e-9: # Ensure there are non-zero values and weights
    # We need to create bins based on the range of y_is_nonzero
    log_y_is_nonzero = np.log10(y_is_nonzero[y_is_nonzero > 1e-9])
    if len(log_y_is_nonzero) > 0:
        min_log_y_is = np.min(log_y_is_nonzero)
        max_log_y_is = np.max(log_y_is_nonzero)
        bins_log_is = np.logspace(min_log_y_is, max_log_y_is, 50)

        # Plot weighted histogram on a log x-scale
        # The weights for density=True should be such that sum(weights_nonzero_norm) = 1
        # hist will normalize so sum(height * bin_width) = 1 if density=True
        # and weights are interpreted as "counts" before normalization.
        axes[1].hist(y_is_nonzero, bins=bins_log_is, weights=weights_nonzero, density=True, alpha=0.7, color='green', label=f'IS P(y|y>0) ({len(y_is_nonzero)} samples)')
        # For density=True with weights, matplotlib normalizes so that the integral of the histogram is 1.
        # This gives the shape of P(y|y>0, using IS samples)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title(f'Distribution of y > 0 (Importance Sampling)\nNote: Weighted P(y=0) ≈ {prop_y_eq_0_is:.2f} (using IS formula)')
        axes[1].set_xlabel('y values (log scale)')
        axes[1].set_ylabel('Density (log scale)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No non-zero y_is samples with positive weight to plot.', ha='center', va='center')
else:
    axes[1].text(0.5, 0.5, 'No non-zero y_is samples or sum of weights is zero.', ha='center', va='center')
axes[1].grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.show()

# --- Alternative plotting with Seaborn for KDE (smoother plot) ---
# This requires careful handling of weights and the log scale for KDE.
# Seaborn's kdeplot can take weights directly.

fig_kde, axes_kde = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Share x-axis for direct comparison

# MC KDE plot
if len(y_mc_nonzero) > 0:
    sns.kdeplot(y_mc_nonzero, ax=axes_kde[0], log_scale=True, color='blue', fill=True, alpha=0.5, label='MC KDE of y (y>0)')
    axes_kde[0].set_title(f'KDE of y > 0 (Standard MC)\nNote: P(y=0) ≈ {len(y_mc_zero)/len(y_mc):.2f}')
    axes_kde[0].set_ylabel('Density')
    axes_kde[0].legend()
else:
    axes_kde[0].text(0.5, 0.5, 'No non-zero y_mc samples for KDE.', ha='center', va='center')
axes_kde[0].grid(True, which="both", ls="-", alpha=0.5)


# IS KDE plot
if len(y_is_nonzero) > 0 and np.sum(weights_nonzero) > 1e-9:
    # For KDE with weights, ensure weights sum to 1 for proper density interpretation if plotting conditional
    # Seaborn handles normalization internally when `weights` are provided.
    sns.kdeplot(y_is_nonzero, weights=weights_nonzero, ax=axes_kde[1], log_scale=True, color='green', fill=True, alpha=0.5, label='IS KDE of y (y>0)')
    axes_kde[1].set_title(f'KDE of y > 0 (Importance Sampling)\nNote: Weighted P(y=0) ≈ {prop_y_eq_0_is:.2f}')
    axes_kde[1].set_xlabel('y values (log scale)')
    axes_kde[1].set_ylabel('Density')
    axes_kde[1].legend()
else:
    axes_kde[1].text(0.5, 0.5, 'No non-zero y_is samples or sum of weights zero for KDE.', ha='center', va='center')
axes_kde[1].grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.show()





#%% Imports
import numpy as np
import pandas as pd # plotnine requires pandas
from scipy import integrate
from scipy.stats import uniform, norm, truncnorm
import matplotlib.pyplot as plt # Still useful for general plot display control maybe

# Import plotnine components
from plotnine import ggplot, aes, geom_histogram, geom_density, scale_x_log10, scale_y_log10
from plotnine import facet_wrap, labs, theme_minimal, theme_matplotlib, after_stat

#%% Range for x
x_min = 0
x_max = 20
num_samples_mc = 100_000
num_samples_is = 10_000 # IS might need fewer samples

#%% Function to calculate y
def calc(x):
    x = np.asarray(x)
    # return np.where((x >= 5) & (x <= 10), (x**3 + 4**x), 0)
    return np.where((x >= 5) & (x <= 10), norm(7.5, 3).pdf(x), 0)

#%% Real expected value of y (for reference)
pdf_p_x_constant = 1.0 / (x_max - x_min)
integrand_real = lambda x_val: calc(x_val) * pdf_p_x_constant
y_exp_real = integrate.quad(integrand_real, x_min, x_max)[0]
print(f"Real expected value of y: {y_exp_real:.6f}")

# Probabilities for y=0 and y>0
prob_y_eq_0 = ((5 - x_min) + (x_max - 10)) / (x_max - x_min)
prob_y_gt_0 = (10 - 5) / (x_max - x_min)
print(f"Theoretical P(y=0): {prob_y_eq_0:.4f}")
print(f"Theoretical P(y>0): {prob_y_gt_0:.4f}")

#%% Standard Monte Carlo
print("\n--- Standard Monte Carlo ---")
p_dist = uniform(loc=x_min, scale=x_max - x_min)
x_mc = p_dist.rvs(num_samples_mc)
y_mc = calc(x_mc)
y_exp_mc = np.mean(y_mc)
print(f"MC expected value of y ({num_samples_mc} samples): {y_exp_mc:.6f}")

# Separate y_mc values that are zero and non-zero
y_mc_nonzero = y_mc[y_mc > 0]
mc_prop_y_eq_0 = 1.0 - len(y_mc_nonzero) / len(y_mc)
print(f"MC: Estimated P(y=0): {mc_prop_y_eq_0:.4f}")

#%% Importance Sampling
print("\n--- Importance Sampling ---")
mu_q = 8.5  # Adjusted mean
sigma_q = 1.5 # Std dev
a_trunc = (x_min - mu_q) / sigma_q
b_trunc = (x_max - mu_q) / sigma_q
q_dist = truncnorm(a=a_trunc, b=b_trunc, loc=mu_q, scale=sigma_q)

x_is = q_dist.rvs(num_samples_is)
y_is = calc(x_is)

p_pdf_values = p_dist.pdf(x_is)
q_pdf_values = q_dist.pdf(x_is)
weights = np.zeros_like(q_pdf_values)
valid_indices = (q_pdf_values > 1e-15) # Use a slightly larger threshold maybe
weights[valid_indices] = p_pdf_values[valid_indices] / q_pdf_values[valid_indices]
weights[~valid_indices] = 0

y_exp_is = np.mean(y_is * weights) # Or np.sum(y_is * weights) / np.sum(weights) if not normalized? Check IS formula. np.mean assumes weights are like frequencies. Correct is sum(y*w)/sum(w) or mean(y*w) if sum(w)=N
# Let's use the standard IS estimator form:
y_exp_is_correct = np.sum(y_is * weights) / num_samples_is # This assumes E[w] = 1
# Or safer: E_q[y * w]
y_exp_is_mean = np.mean(y_is * weights)

print(f"IS expected value of y ({num_samples_is} samples, mu_q={mu_q}, sigma_q={sigma_q}) (using np.mean): {y_exp_is_mean:.6f}")

# Separate y_is values and their weights
y_is_nonzero_indices = (y_is > 0)
y_is_nonzero = y_is[y_is_nonzero_indices]
weights_nonzero = weights[y_is_nonzero_indices]

# Estimate P(y=0) using IS weights
is_prop_y_eq_0 = 0.0
total_weight = np.sum(weights)
if total_weight > 1e-9:
    is_prop_y_eq_0 = np.sum(weights[~y_is_nonzero_indices]) / total_weight
    print(f"IS: Weighted P(y=0): {is_prop_y_eq_0:.4f}")
else:
    print("IS: Sum of weights too small to estimate P(y=0).")


#%% Prepare DataFrames for plotnine

# Create MC DataFrame
df_mc = pd.DataFrame({
    'y_value': y_mc_nonzero,
    'method': 'MC',
    'weight': 1.0 # Each MC sample has equal weight
})

# Create IS DataFrame
df_is = pd.DataFrame({
    'y_value': y_is_nonzero,
    'method': 'IS',
    'weight': weights_nonzero # Use the calculated importance weights
})

# Combine DataFrames
df_plot = pd.concat([df_mc, df_is], ignore_index=True)

# Filter out potentially problematic zero/negative values before log scaling if any slipped through
df_plot = df_plot[df_plot['y_value'] > 1e-9]

#%% Create plots using plotnine

# --- Histogram Plot ---
# Define dynamic titles based on calculated probabilities
title_hist = (f"Histogram of y > 0 (Conditional Density P(y|y>0))\n"
              f"Est. P(y=0): MC={mc_prop_y_eq_0:.2f}, IS(weighted)={is_prop_y_eq_0:.2f}")

plot_hist = (
    ggplot(df_plot, aes(x='y_value', weight='weight', fill='method'))
    + geom_histogram(aes(y=after_stat('density')), # Use after_stat for density
                     alpha=0.7, bins=50, position='identity') # Use identity for overlay potential, fine for facet
    + scale_x_log10(name="y value (log scale)")
    + scale_y_log10(name="Density P(y|y>0) (log scale)") # Log scale for density too
    + facet_wrap('~method', scales='fixed') # Separate plots for MC and IS, fixed scales aid comparison
    + labs(title=title_hist, fill="Sampling Method")
    # + theme_minimal()
    # + theme_matplotlib() # Alternative theme
)

# --- Density Plot (KDE) ---
title_kde = (f"KDE of y > 0 (Conditional Density P(y|y>0))\n"
             f"Est. P(y=0): MC={mc_prop_y_eq_0:.2f}, IS(weighted)={is_prop_y_eq_0:.2f}")

plot_kde = (
    ggplot(df_plot, aes(x='y_value', weight='weight', color='method'))
    + geom_density(aes(fill='method'), alpha=0.5) # Color lines, fill areas
    + scale_x_log10(name="y value (log scale)")
    # Density plots might look better without log scale on y-axis, or need careful limits
    + scale_y_log10(name="Density P(y|y>0) (log scale)", limits=(1e-10, None)) # Add small lower limit for log scale
    # + scale_y_continuous(name="Density P(y|y>0)") # Alternative: linear y-scale
    + facet_wrap('~method', scales='fixed')
    + labs(title=title_kde, color="Sampling Method", fill="Sampling Method")
    # + theme_minimal()
)

#%%
print (plot_hist)
# print (plot_kde)

#%% Display the plots
print("\nDisplaying plotnine histograms...")
plot_hist.draw()
plt.show() # Might be needed depending on environment

print("\nDisplaying plotnine density plots...")
plot_kde.draw()
plt.show() # Might be needed depending on environment


#endregion -----------------------------------------------------------------------------------------
#region Random Tests

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
#region Sampling Tests

#%% Libraries
import dsplus as ds
from modules.compute_raster_stats import match_crs_to_raster
from modules.shift_storm_center import shift_gdf
from modules.compute_raster_stats import sum_raster_values_in_polygon
from modules.compute_depths import compute_depths, print_sim_stats, get_df_freq_curve

#%% Data
os.chdir(r'D:\FEMA Innovations\SO3.1\Py\Trinity')
path_storm = pathlib.Path('storm_catalogue_trinity')
path_sp_watershed = r"D:\FEMA Innovations\SO3.1\Py\Trinity\watershed\trinity.geojson"
path_sp_domain = r"D:\FEMA Innovations\SO3.1\Py\Trinity\watershed\trinity-transpo-area-v01.geojson"
df_storms = pd.read_pickle(path_storm/'catalogue.pkl')
sp_watershed = gpd.read_file(path_sp_watershed)
sp_domain = gpd.read_file(path_sp_domain)
sp_watershed = match_crs_to_raster(sp_watershed, df_storms['path'].iloc[0])
sp_domain = match_crs_to_raster(sp_domain, df_storms['path'].iloc[0])
df_storm_sample_mc_0: pd.DataFrame = pd.read_pickle('df_storm_sample_mc_0.pkl')
df_storm_sample_mc_1: pd.DataFrame = pd.read_pickle('df_storm_sample_mc_1.pkl')
df_depths_mc_0: pd.DataFrame = pd.read_pickle('df_depths_mc_0.pkl')
df_depths_mc_1: pd.DataFrame = pd.read_pickle('df_depths_mc_1.pkl')
choice_dist = 'TruncNorm'
choice_param_value = 1.2
choice_param_name = 'std'
# choice_dist = 'TruncGenNorm'
# choice_param_value = 5
# choice_param_name = 'beta'
if choice_dist == 'TruncNorm':
    mult_std = choice_param_value
    df_storm_sample_is_1 = pd.read_pickle(f'df_storm_sample_is_tn_std_{mult_std}.pkl')
    df_depths_is_1 = pd.read_pickle(f'df_depths_is_tn_std_{mult_std}.pkl')
else:
    beta = choice_param_value
    df_storm_sample_is_1 = pd.read_pickle(f'df_storm_sample_is_tgn_beta_{beta}.pkl')
    df_depths_is_1 = pd.read_pickle(f'df_depths_is_tgn_beta_{beta}.pkl')


#%% Centroid file
sp_centroid = ds.sp_points_from_df_xy(df_storms, crs=sp_watershed.crs)
sp_centroid.to_file('storm_centroid.shp')

#%%
i = 0
_df_storm_sample_mc_0 = df_storm_sample_mc_0.iloc[[i]]
row = df_depths_mc_0.iloc[i]

sp_centroid_sample = ds.sp_points_from_df_xy(df_storms.loc[lambda _: _.name == row['name']], crs=sp_watershed.crs)
sp_centroid_sample.to_file('_sample_sp_centroid_sample.shp')

sp_centroid_sample_shifted = ds.sp_points_from_df_xy(_df_storm_sample_mc_0, column_x='x_sampled', column_y='y_sampled', crs=sp_watershed.crs)
sp_centroid_sample_shifted.to_file('_sample_sp_centroid_sample_shifted.shp')

sp_watershed_shifted = shift_gdf(sp_watershed, -row.x_del, -row.y_del)
sp_watershed_shifted.to_file('_sample_sp_watershed_shifted.shp')

row['depth']
sum_raster_values_in_polygon(row.path, sp_watershed_shifted)

#%%
df_storm_sample = df_storm_sample_mc_0.sample(1500)
df_1 = compute_depths(df_storm_sample, sp_watershed)
df_2 = compute_depths(df_storm_sample, sp_watershed, parallel=True)

df_1.equals(df_2)

#endregion -----------------------------------------------------------------------------------------
