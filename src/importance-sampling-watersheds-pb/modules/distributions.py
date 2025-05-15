#region Libraries

#%%
import numpy as np
from scipy.stats import gennorm

#endregion -----------------------------------------------------------------------------------------
#region Distributions

#%% Function to pass parameters to truncnorm
def truncnorm_params(mean: float, std_dev: float, lower: float, upper: float) -> dict:
    '''
    Calculate the parameters for a truncated normal distribution.

    Args:
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
        lower (float): The lower bound of the truncated distribution.
        upper (float): The upper bound of the truncated distribution.

    Returns:
        dict: A dictionary containing the parameters 'a', 'b', 'loc', and 'scale' for the truncated normal distribution.

    '''
    d = dict(
        a = (lower - mean) / std_dev,
        b = (upper - mean) / std_dev,
        loc = mean,
        scale = std_dev
    )

    return d

#%% Truncated Generalized Normal Distribution
class TruncatedGeneralizedNormal:
    '''A truncated generalized normal distribution.

    Args:
        loc (float): Location Parameter.
        scale (float): Scale Parameter.
        beta (float): Shape Parameter.
        lower_bound (float): Lower truncation bound.
        upper_bound (float): Upper truncation bound.
    '''
    def __init__(self, beta: float, loc: float, scale: float, lower_bound: float, upper_bound: float):
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

#endregion -----------------------------------------------------------------------------------------
