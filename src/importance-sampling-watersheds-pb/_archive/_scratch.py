#region Libraries

#%%
import geopandas as gpd
import os
import pathlib

#endregion -----------------------------------------------------------------------------------------
#region Functions (Mixtures)

#%%
import numpy as np
import scipy.stats as stats
import scipy
import scipy.optimize  # For brentq (root finding for PPF)
import matplotlib.pyplot as plt


#%%
class TruncatedDistribution:
    """
    A class to create a truncated version of a scipy.stats continuous distribution.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        The original continuous distribution object from scipy.stats.
        Example: stats.norm(loc=0, scale=1), stats.expon(scale=2)
    a : float
        The lower bound for truncation.
    b : float
        The upper bound for truncation.

    Methods
    -------
    pdf(x)
        Probability density function.
    cdf(x)
        Cumulative distribution function.
    ppf(q)
        Percent point function (inverse of cdf).
    rvs(size=1, random_state=None)
        Random variates.
    support()
        Returns the support of the distribution (a, b).
    mean()
        Calculates the mean of the truncated distribution (optional, via numerical integration).
    var()
        Calculates the variance of the truncated distribution (optional, via numerical integration).
    """
    def __init__(self, dist, a, b):
        # if not isinstance(dist, stats.rv_continuous):
        #     raise TypeError("dist must be a continuous distribution from scipy.stats.")
        if not np.isfinite(a) or not np.isfinite(b):
            # While possible, let's keep it simple for now.
            # Truncating an already truncated distribution like truncnorm can be tricky.
            # scipy.stats.truncnorm handles infinite bounds better.
            raise ValueError("Truncation bounds 'a' and 'b' must be finite.")
        if a >= b:
            raise ValueError(f"Lower bound 'a' ({a}) must be less than upper bound 'b' ({b}).")

        self.dist = dist
        self.a = float(a)
        self.b = float(b)

        # Pre-calculate CDF values at bounds for efficiency and normalization
        self.cdf_a = self.dist.cdf(self.a)
        self.cdf_b = self.dist.cdf(self.b)
        
        self.prob_interval = self.cdf_b - self.cdf_a

        if self.prob_interval <= 0: # Use a small epsilon if very precise checks are needed
            raise ValueError(
                f"The probability mass of the original distribution "
                f"in the interval [{self.a}, {self.b}] is non-positive "
                f"({self.prob_interval:.2e}). "
                "Check if bounds are too far in the tails or identical."
            )
        
        # For optional moment calculations
        self._moments_calculated = False
        self._mean = None
        self._var = None

    def pdf(self, x):
        """Probability density function."""
        x = np.asarray(x)
        # Initialize pdf values to 0.0
        pdf_values = np.zeros_like(x, dtype=float)
        
        # Identify mask for values within the truncation interval [a, b]
        mask = (x >= self.a) & (x <= self.b)
        
        # Calculate PDF for values within the interval
        if np.any(mask):
            pdf_values[mask] = self.dist.pdf(x[mask]) / self.prob_interval
            
        return pdf_values

    def cdf(self, x):
        """Cumulative distribution function."""
        x = np.asarray(x)
        # Initialize cdf values
        cdf_values = np.zeros_like(x, dtype=float)
        
        # Mask for values less than 'a' (cdf is 0)
        # mask_below_a = x < self.a (already handled by zeros_like)

        # Mask for values greater than 'b' (cdf is 1)
        mask_above_b = x > self.b
        cdf_values[mask_above_b] = 1.0
        
        # Mask for values within the interval [a, b]
        mask_interval = (x >= self.a) & (x <= self.b)
        if np.any(mask_interval):
            cdf_values[mask_interval] = (self.dist.cdf(x[mask_interval]) - self.cdf_a) / self.prob_interval
            
        return cdf_values

    def ppf(self, q):
        """Percent point function (inverse of cdf)."""
        q = np.asarray(q)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Input 'q' (quantiles) must be between 0 and 1.")
            
        # Transform q back to the original distribution's CDF scale
        original_cdf_q = self.cdf_a + q * self.prob_interval
        
        # Use the original distribution's PPF
        # Ensure values are clamped if original_cdf_q slightly outside [cdf_a, cdf_b] due to precision
        # This is generally handled well by ppf if original_cdf_q is within [0,1] for the base dist.
        ppf_values = self.dist.ppf(original_cdf_q)

        # Clamp results to the interval [a,b] to handle potential floating point issues
        # or if the base distribution's ppf gives something slightly outside due to its own numerics.
        # For q=0, we expect a. For q=1, we expect b.
        ppf_values = np.clip(ppf_values, self.a, self.b)
        # Specifically handle edges for perfect 0 and 1
        ppf_values[q == 0] = self.a
        ppf_values[q == 1] = self.b

        return ppf_values

    def rvs(self, size=1, random_state=None):
        """Random variates."""
        if random_state is not None:
            # For reproducibility if a seed/generator is passed
            # If using a global random state, this line is not strictly needed
            # but good practice for isolated components.
            # However, self.dist might use its own random_state if set.
            # This makes it a bit complex. Simplest is to use np.random.
            # For full control, the `dist` object should also be configured
            # with a random_state if its rvs method is used directly.
            # Here, we are using ppf, so np.random.uniform is the source.
            rng = np.random.default_rng(random_state)
            uniform_variates = rng.uniform(low=0.0, high=1.0, size=size)
        else:
            uniform_variates = np.random.uniform(low=0.0, high=1.0, size=size)
            
        return self.ppf(uniform_variates)

    def support(self):
        """Returns the support of the distribution (a, b)."""
        return (self.a, self.b)

    # --- Optional: Mean and Variance ---
    # These require numerical integration if a closed form is not simple.
    # Scipy provides `expect` for this.
    
    def _calculate_moments(self):
        """Helper to calculate mean and variance once."""
        if not self._moments_calculated:
            # Mean: E[X] = integral from a to b of (x * pdf_original(x) / prob_interval) dx
            # Variance: E[X^2] - (E[X])^2
            # E[X^2] = integral from a to b of (x^2 * pdf_original(x) / prob_interval) dx
            
            # Using scipy.stats.rv_continuous.expect
            # The 'func' argument to expect is applied to the random variable X.
            # The expectation is taken with respect to the *original* distribution.
            # E_trunc[g(X)] = E_orig[g(X) * I(a<=X<=b)] / P(a<=X<=b)
            # where I is indicator function.
            # scipy.stats.expect does: integral func(x) * pdf_original(x) dx over bounds
            
            # Mean
            integrand_mean = lambda x: x 
            # Calculate E_orig[X] but only over [a,b]
            # This isn't directly E_trunc[X].
            # We need E_orig[X * I(a<=X<=b)] / prob_interval
            # which is (integral from a to b of x * pdf_orig(x) dx) / prob_interval

            # Scipy's expect function for a distribution `dist` computes:
            # integral func(x) * dist.pdf(x) dx
            # The integration limits are taken from dist.support() unless overridden by `lb`, `ub`.
            
            expected_x_in_interval = self.dist.expect(func=lambda x: x, lb=self.a, ub=self.b)
            self._mean = expected_x_in_interval / self.prob_interval
            
            expected_x_squared_in_interval = self.dist.expect(func=lambda x: x**2, lb=self.a, ub=self.b)
            e_x2_truncated = expected_x_squared_in_interval / self.prob_interval
            
            self._var = e_x2_truncated - (self._mean**2)
            self._moments_calculated = True

    def mean(self):
        """Calculates the mean of the truncated distribution."""
        if not self._moments_calculated:
            self._calculate_moments()
        return self._mean

    def var(self):
        """Calculates the variance of the truncated distribution."""
        if not self._moments_calculated:
            self._calculate_moments()
        return self._var

    def __repr__(self):
        return f"TruncatedDistribution(a={self.a}, b={self.b})"

#%%
class MixtureDistribution:
    """
    A class to create a two-component mixture of scipy.stats continuous distributions.

    Parameters
    ----------
    dist1 : scipy.stats.rv_continuous
        The first continuous distribution object (frozen, i.e., with parameters set).
        Example: stats.norm(loc=0, scale=1)
    dist2 : scipy.stats.rv_continuous
        The second continuous distribution object (frozen).
        Example: stats.expon(scale=2)
    weight1 : float
        The weight for the first distribution (dist1). Must be between 0 and 1.
        The weight for the second distribution (dist2) will be (1 - weight1).

    Methods
    -------
    pdf(x)
        Probability density function.
    cdf(x)
        Cumulative distribution function.
    ppf(q)
        Percent point function (inverse of cdf). Requires numerical solving.
    rvs(size=1, random_state=None)
        Random variates.
    mean()
        Calculates the mean of the mixture distribution.
    var()
        Calculates the variance of the mixture distribution.
    support()
        Returns the support of the mixture distribution.
    """

    def __init__(self, dist1, dist2, weight1):
        if not (hasattr(dist1, 'pdf') and hasattr(dist1, 'cdf') and hasattr(dist1, 'ppf') and hasattr(dist1, 'rvs') and
                hasattr(dist2, 'pdf') and hasattr(dist2, 'cdf') and hasattr(dist2, 'ppf') and hasattr(dist2, 'rvs')):
            raise TypeError(
                "dist1 and dist2 must be frozen distribution objects from scipy.stats "
                "with pdf, cdf, ppf, and rvs methods."
            )
        # A simple check for continuous type based on common practice for scipy.stats
        if not (isinstance(dist1.dist, stats.rv_continuous) and isinstance(dist2.dist, stats.rv_continuous)):
             # This check might be too strict if a custom object mimicking rv_continuous is used.
             # A more duck-typing approach would be to check for methods.
             print("Warning: One or both distributions might not be rv_continuous from scipy.stats.")


        if not (0 <= weight1 <= 1):
            raise ValueError("weight1 must be between 0 and 1 (inclusive).")

        self.dist1 = dist1
        self.dist2 = dist2
        self.weight1 = float(weight1)
        self.weight2 = 1.0 - self.weight1
        
        d1_name = getattr(getattr(self.dist1, 'dist', self.dist1), 'name', 'dist1')
        d2_name = getattr(getattr(self.dist2, 'dist', self.dist2), 'name', 'dist2')
        self.name = f"Mixture({self.weight1:.2f}*{d1_name}, {self.weight2:.2f}*{d2_name})"


        # Determine support of the mixture
        s1_a, s1_b = self.dist1.support()
        s2_a, s2_b = self.dist2.support()
        self._support_a = min(s1_a, s2_a)
        self._support_b = max(s1_b, s2_b)

        # Heuristic bounds for PPF root finding. These are wide fallbacks.
        # Prefer support if finite, otherwise use component ppfs at small/large quantiles.
        eps = 1e-9 # Small epsilon for ppf bounds
        
        candidate_lows = [self._support_a] # Start with support
        d1_ppf_eps = self.dist1.ppf(eps)
        if np.isfinite(d1_ppf_eps): candidate_lows.append(d1_ppf_eps)
        d2_ppf_eps = self.dist2.ppf(eps)
        if np.isfinite(d2_ppf_eps): candidate_lows.append(d2_ppf_eps)
        self._ppf_search_lower = max(filter(np.isfinite, candidate_lows), default=-1e14)


        candidate_highs = [self._support_b] # Start with support
        d1_ppf_1eps = self.dist1.ppf(1-eps)
        if np.isfinite(d1_ppf_1eps): candidate_highs.append(d1_ppf_1eps)
        d2_ppf_1eps = self.dist2.ppf(1-eps)
        if np.isfinite(d2_ppf_1eps): candidate_highs.append(d2_ppf_1eps)
        self._ppf_search_upper = min(filter(np.isfinite, candidate_highs), default=1e14)

        # Ensure lower < upper for search bounds
        if self._ppf_search_lower >= self._ppf_search_upper:
            # This can happen if supports are single points or very narrow and eps ppfs cross over.
            # Use a minimal sensible range around the support.
            mid_support = (self._support_a + self._support_b) / 2.0
            if np.isfinite(mid_support):
                self._ppf_search_lower = mid_support - 1.0
                self._ppf_search_upper = mid_support + 1.0
            else: # Both supports inf or -inf, use wide defaults
                self._ppf_search_lower = -1e14
                self._ppf_search_upper = 1e14

    def pdf(self, x):
        """Probability density function."""
        x = np.asarray(x)
        pdf_val = (self.weight1 * self.dist1.pdf(x) +
                   self.weight2 * self.dist2.pdf(x))
        return pdf_val

    def cdf(self, x):
        """Cumulative distribution function."""
        x = np.asarray(x)
        cdf_val = (self.weight1 * self.dist1.cdf(x) +
                   self.weight2 * self.dist2.cdf(x))
        return cdf_val

    def _solve_ppf_scalar(self, q_scalar):
        """Helper to solve PPF for a single scalar q using numerical root finding."""
        if not (0 <= q_scalar <= 1):
            # This should be caught by the public ppf method, but good to have.
            return np.nan 
        
        if np.isclose(q_scalar, 0.0):
            return self._support_a
        if np.isclose(q_scalar, 1.0):
            return self._support_b

        func_to_solve = lambda x_val: self.cdf(x_val) - q_scalar

        # Initial search bracket
        # Use pre-calculated general bounds, refine if possible
        a, b = self._ppf_search_lower, self._ppf_search_upper

        # Attempt to make the bracket tighter and ensure it straddles the root
        # Try component PPFs as hints for the bracket
        ppf_vals = []
        try: ppf_vals.append(self.dist1.ppf(q_scalar))
        except: pass # Catch errors if ppf fails for q_scalar
        try: ppf_vals.append(self.dist2.ppf(q_scalar))
        except: pass
        
        finite_ppf_vals = [v for v in ppf_vals if np.isfinite(v)]
        if finite_ppf_vals:
            # Tentative bracket based on component ppfs
            min_ppf = min(finite_ppf_vals)
            max_ppf = max(finite_ppf_vals)
            # Expand this tentative bracket slightly
            delta = max(1.0, abs(max_ppf - min_ppf) * 0.1) 
            current_a = min_ppf - delta
            current_b = max_ppf + delta
            
            # Use this refined bracket if it's within the global search bounds
            a = max(a, current_a)
            b = min(b, current_b)

        # Ensure a < b, if not, widen them
        if a >= b:
            a -= 1.0 # fallback to make them distinct
            b += 1.0
            if a >=b: # extreme case, support is likely a single point or very narrow
                a = self._support_a -1e-6
                b = self._support_b +1e-6


        # Brentq requires f(a) and f(b) to have opposite signs.
        # Adjust 'a' downwards if f(a) >= 0
        fa = func_to_solve(a)
        max_shrink_iters = 10
        iter_count = 0
        while fa >= 0 and iter_count < max_shrink_iters:
            if np.isclose(fa,0): break # a is the root
            a_new = a - max(1.0, abs(a * 0.5)) # Try to move 'a' significantly lower
            if not np.isfinite(a_new) or a_new < self._support_a - 1e7: # Safety break for extreme values
                a = max(self._support_a, -1e15) # Fallback to support or very large neg number
                break
            a = a_new
            fa = func_to_solve(a)
            iter_count += 1
        
        # Adjust 'b' upwards if f(b) <= 0
        fb = func_to_solve(b)
        iter_count = 0
        while fb <= 0 and iter_count < max_shrink_iters:
            if np.isclose(fb,0): break # b is the root
            b_new = b + max(1.0, abs(b * 0.5)) # Try to move 'b' significantly higher
            if not np.isfinite(b_new) or b_new > self._support_b + 1e7: # Safety break
                b = min(self._support_b, 1e15) # Fallback to support or very large pos number
                break
            b = b_new
            fb = func_to_solve(b)
            iter_count +=1

        try:
            if np.isclose(fa,0): return a
            if np.isclose(fb,0): return b
            return scipy.optimize.brentq(func_to_solve, a, b, xtol=1e-9, rtol=1e-9, maxiter=100)
        except ValueError:
            # This typically means f(a) and f(b) don't have opposite signs or other brentq issue
            # print(f"Warning: brentq failed for q={q_scalar:.4f}. Bounds [{a:.2e}, {b:.2e}], f(a)={fa:.2e}, f(b)={fb:.2e}.")
            # Fallback or raise error
            # A simple fallback could be nan or a weighted average of component ppfs (crude)
            if finite_ppf_vals:
                return self.weight1 * self.dist1.ppf(q_scalar) + self.weight2 * self.dist2.ppf(q_scalar) if len(finite_ppf_vals)==2 else finite_ppf_vals[0]
            return np.nan


    def ppf(self, q):
        """Percent point function (inverse of cdf)."""
        q_arr = np.asarray(q)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("Input 'q' (quantiles) must be between 0 and 1.")

        if q_arr.ndim == 0: # Scalar input
            return self._solve_ppf_scalar(q_arr.item())
        else: # Array input
            results = np.empty_like(q_arr, dtype=float)
            for i, q_scalar_val in np.ndenumerate(q_arr):
                results[i] = self._solve_ppf_scalar(q_scalar_val)
            return results

    def rvs(self, size=1, random_state=None):
        """Random variates."""
        rng = np.random.default_rng(random_state)
        
        # Determine which distribution to sample from for each variate
        u_choices = rng.uniform(size=size)
        from_dist1_mask = u_choices < self.weight1
        
        num_from_dist1 = np.sum(from_dist1_mask)
        num_from_dist2 = size - num_from_dist1
        
        samples = np.empty(size, dtype=float)
        
        if num_from_dist1 > 0:
            samples[from_dist1_mask] = self.dist1.rvs(size=num_from_dist1, random_state=rng)
        
        if num_from_dist2 > 0:
            # Ensure we select the correct elements for dist2 samples
            samples[~from_dist1_mask] = self.dist2.rvs(size=num_from_dist2, random_state=rng)
            
        return samples

    def mean(self):
        """Calculates the mean of the mixture distribution."""
        mean1 = self.dist1.mean()
        mean2 = self.dist2.mean()
        if np.isnan(mean1) or np.isnan(mean2): # If any component mean is NaN (e.g., Cauchy)
            return np.nan
        return self.weight1 * mean1 + self.weight2 * mean2

    def var(self):
        """Calculates the variance of the mixture distribution."""
        mean1 = self.dist1.mean()
        var1 = self.dist1.var()
        
        mean2 = self.dist2.mean()
        var2 = self.dist2.var()

        if np.isnan(mean1) or np.isnan(var1) or np.isnan(mean2) or np.isnan(var2):
            return np.nan # If any component moment is NaN

        # E[X^2] = w1 * E[X1^2] + w2 * E[X2^2]
        # E[Xi^2] = Var(Xi) + (E[Xi])^2
        e_x1_sq = var1 + mean1**2
        e_x2_sq = var2 + mean2**2
        
        mixture_e_x_sq = self.weight1 * e_x1_sq + self.weight2 * e_x2_sq
        
        # Use self.mean() to avoid re-calculating means if they were stored,
        # but here it's fine to recalculate or call self.mean()
        current_mixture_mean = self.mean()
        if np.isnan(current_mixture_mean): # Check if mean itself is NaN
            return np.nan

        mixture_variance = mixture_e_x_sq - current_mixture_mean**2
        return mixture_variance
        
    def support(self):
        """Returns the support of the mixture distribution (min_support, max_support)."""
        return (self._support_a, self._support_b)

    def __repr__(self):
        d1_repr = getattr(self.dist1, '__repr__', lambda: str(self.dist1))()
        d2_repr = getattr(self.dist2, '__repr__', lambda: str(self.dist2))()
        return (f"MixtureDistribution(dist1={d1_repr}, \n"
                f"                    dist2={d2_repr}, \n"
                f"                    weight1={self.weight1:.3f})")

#%% --- Example Usage ---
if __name__ == "__main__":
    # 1. Define an original distribution
    original_norm = stats.norm(loc=5, scale=2) # Normal distribution with mean 5, std 2

    # 2. Define truncation bounds
    a_trunc = 3.0
    b_trunc = 8.0

    # 3. Create the truncated distribution
    try:
        truncated_norm = TruncatedDistribution(original_norm, a_trunc, b_trunc)
        print(f"Created: {truncated_norm}")

        # 4. Test its methods
        x_values = np.linspace(a_trunc - 1, b_trunc + 1, 500)
        
        # PDF
        pdf_vals = truncated_norm.pdf(x_values)
        
        # CDF
        cdf_vals = truncated_norm.cdf(x_values)
        
        # PPF
        q_values = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        ppf_vals = truncated_norm.ppf(q_values)
        print(f"\nPPF for q={q_values}:")
        for q, p_val in zip(q_values, ppf_vals):
            print(f"  PPF({q:.2f}) = {p_val:.4f}")

        # RVS
        num_samples = 10000
        samples = truncated_norm.rvs(size=num_samples, random_state=42)
        print(f"\nGenerated {num_samples} samples. Shape: {samples.shape}")
        print(f"Sample min: {samples.min():.4f}, Sample max: {samples.max():.4f}")
        print(f"Sample mean: {np.mean(samples):.4f}, Sample std: {np.std(samples):.4f}")

        # Mean and Variance from class methods
        print(f"\nTheoretical mean (truncated): {truncated_norm.mean():.4f}")
        print(f"Theoretical variance (truncated): {truncated_norm.var():.4f}")
        print(f"Theoretical std (truncated): {np.sqrt(truncated_norm.var()):.4f}")
        print(f"Support: {truncated_norm.support()}")

        # 5. Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False) # sharex=False as PPF has different x-axis

        # Plot PDF
        axs[0].plot(x_values, pdf_vals, 'r-', lw=2, label='Truncated PDF')
        axs[0].plot(x_values, original_norm.pdf(x_values), 'b--', lw=1, alpha=0.7, label='Original PDF')
        axs[0].fill_between(x_values, pdf_vals, where=(x_values >= a_trunc) & (x_values <= b_trunc), color='red', alpha=0.3)
        axs[0].axvline(a_trunc, color='gray', linestyle=':', label=f'a={a_trunc}')
        axs[0].axvline(b_trunc, color='gray', linestyle=':', label=f'b={b_trunc}')
        axs[0].set_title(f'PDF of Truncated Normal ({original_norm.args}, a={a_trunc}, b={b_trunc})')
        axs[0].set_ylabel('Density')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # Plot CDF
        axs[1].plot(x_values, cdf_vals, 'r-', lw=2, label='Truncated CDF')
        axs[1].plot(x_values, original_norm.cdf(x_values), 'b--', lw=1, alpha=0.7, label='Original CDF')
        axs[1].axvline(a_trunc, color='gray', linestyle=':')
        axs[1].axvline(b_trunc, color='gray', linestyle=':')
        axs[1].set_title('CDF of Truncated Normal')
        axs[1].set_ylabel('Cumulative Probability')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot PPF points and histogram of RVS
        # For PPF, plot q_values vs ppf_vals
        axs[2].plot(q_values, ppf_vals, 'go-', label='Truncated PPF values')
        # Overlay histogram of generated samples
        axs[2].hist(samples, bins=50, density=True, alpha=0.6, label=f'{num_samples} RVS samples (hist)')
        # For comparison, plot the PDF scaled appropriately if desired (might clutter)
        # x_for_pdf_overlay = np.linspace(a_trunc, b_trunc, 200)
        # axs[2].plot(x_for_pdf_overlay, truncated_norm.pdf(x_for_pdf_overlay), 'r--', alpha=0.8, label='Truncated PDF (for hist scale)')
        axs[2].set_title('PPF and Histogram of RVS')
        axs[2].set_xlabel('Quantile (q) for PPF / Value for Histogram')
        axs[2].set_ylabel('PPF Value / Density for Histogram')
        axs[2].legend()
        axs[2].grid(True, linestyle='--', alpha=0.7)


        plt.tight_layout()
        plt.show()

        # Example with another distribution: Exponential
        original_expon = stats.expon(scale=10) # Mean of exponential is its scale
        a_expon, b_expon = 5, 20
        truncated_expon = TruncatedDistribution(original_expon, a_expon, b_expon)
        print(f"\nCreated: {truncated_expon}")
        print(f"Mean of truncated exponential: {truncated_expon.mean():.4f}")
        
        samples_expon = truncated_expon.rvs(1000)
        plt.figure(figsize=(8,5))
        plt.hist(samples_expon, bins=30, density=True, alpha=0.7, label='Truncated Exponential RVS')
        x_plot = np.linspace(a_expon, b_expon, 200)
        plt.plot(x_plot, truncated_expon.pdf(x_plot), 'r-', label='Truncated Exponential PDF')
        plt.title(f'Truncated Exponential (scale=10, a={a_expon}, b={b_expon})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    except ValueError as e:
        print(f"Error creating or using TruncatedDistribution: {e}")
    except TypeError as e:
        print(f"Type Error: {e}")

#%% --- Example Usage ---
if __name__ == "__main__":
    # 1. Define component distributions (must be "frozen" with parameters set)
    # Bimodal normal mixture
    dist_comp1 = stats.norm(loc=-3, scale=1.0) 
    dist_comp2 = stats.norm(loc=3, scale=1.5)
    weight_1 = 0.4

    # Normal and Exponential mixture
    # dist_comp1 = stats.norm(loc=0, scale=2)
    # dist_comp2 = stats.expon(loc=1, scale=3) # loc is shift, scale is 1/lambda (mean for expon is loc+scale)
    # weight_1 = 0.6

    # 2. Create the mixture distribution
    mixture_dist = MixtureDistribution(dist_comp1, dist_comp2, weight_1)
    print(mixture_dist)
    print(f"Support of mixture: {mixture_dist.support()}")

    # 3. Test its methods
    # Define a range for plotting based on significant probability mass
    plot_low = mixture_dist.ppf(0.0001)
    plot_high = mixture_dist.ppf(0.9999)
    if not (np.isfinite(plot_low) and np.isfinite(plot_high) and plot_low < plot_high) : # Fallback if ppf failed
        plot_low = -10
        plot_high = 10
    x_values = np.linspace(plot_low, plot_high, 500)
    
    pdf_values = mixture_dist.pdf(x_values)
    cdf_values = mixture_dist.cdf(x_values)
    
    print(f"\nTheoretical mean: {mixture_dist.mean():.4f}")
    print(f"Theoretical variance: {mixture_dist.var():.4f}")
    print(f"Theoretical std dev: {np.sqrt(mixture_dist.var()):.4f}")

    # Test PPF
    q_for_ppf = np.array([0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0])
    ppf_values_test = mixture_dist.ppf(q_for_ppf)
    print("\nPPF Test:")
    for q_val, x_ppf in zip(q_for_ppf, ppf_values_test):
        if np.isfinite(x_ppf):
            cdf_check = mixture_dist.cdf(x_ppf)
            print(f"  PPF({q_val:.3f}) = {x_ppf:9.4f}  => CDF(x_ppf) = {cdf_check:.4f} (Error: {cdf_check-q_val:.2e})")
        else:
            print(f"  PPF({q_val:.3f}) = {x_ppf:9.4f}")


    # Generate random samples
    num_rvs_samples = 20000
    rvs_samples = mixture_dist.rvs(size=num_rvs_samples, random_state=42)
    print(f"\nGenerated {num_rvs_samples} samples:")
    print(f"  Sample mean: {np.mean(rvs_samples):.4f}")
    print(f"  Sample std dev: {np.std(rvs_samples):.4f}")

    # 4. Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Plot PDF and histogram of RVS
    axs[0].plot(x_values, pdf_values, 'r-', lw=2, label='Mixture PDF')
    axs[0].hist(rvs_samples, bins=100, density=True, alpha=0.6, label='RVS Histogram')
    # Overlay component PDFs (weighted)
    axs[0].plot(x_values, weight_1 * dist_comp1.pdf(x_values), 'g:', lw=1.5, alpha=0.7, label=f'w1*Comp1 PDF')
    axs[0].plot(x_values, (1-weight_1) * dist_comp2.pdf(x_values), 'b:', lw=1.5, alpha=0.7, label=f'w2*Comp2 PDF')
    axs[0].set_title(f'Mixture PDF and RVS\n{mixture_dist.name}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Density')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot CDF
    axs[1].plot(x_values, cdf_values, 'r-', lw=2, label='Mixture CDF')
    # Overlay component CDFs (weighted contribution, not true CDFs of components)
    # axs[1].plot(x_values, weight_1 * dist_comp1.cdf(x_values), 'g:', lw=1, alpha=0.7, label=f'w1*Comp1 CDF term')
    # axs[1].plot(x_values, (1-weight_1) * dist_comp2.cdf(x_values), 'b:', lw=1, alpha=0.7, label=f'w2*Comp2 CDF term')
    axs[1].set_title('Mixture CDF')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Plot PPF
    axs[2].plot(q_for_ppf, ppf_values_test, 'mo-', markersize=8, label='Mixture PPF values (q vs x)')
    axs[2].plot(cdf_values, x_values, 'c--', alpha = 0.5, label='Mixture CDF (inverted, for visual check)') # Plotting (CDF, x) should match (q, PPF(q))
    axs[2].set_title('Mixture PPF')
    axs[2].set_xlabel('Quantile (q)')
    axs[2].set_ylabel('Value (x)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # Test with Cauchy (undefined mean/var)
    print("\n--- Testing with Cauchy component (undefined mean/var) ---")
    cauchy_dist = stats.cauchy(loc=0, scale=0.5)
    norm_dist_for_cauchy_mix = stats.norm(loc=5, scale=1)
    try:
        mixture_with_cauchy = MixtureDistribution(cauchy_dist, norm_dist_for_cauchy_mix, 0.3)
        print(mixture_with_cauchy)
        print(f"Support: {mixture_with_cauchy.support()}") # Should be (-inf, inf)
        print(f"Mean: {mixture_with_cauchy.mean()}") # Should be NaN
        print(f"Variance: {mixture_with_cauchy.var()}") # Should be NaN
        
        # Test PPF for Cauchy mixture
        q_cauchy_test = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        ppf_cauchy_results = mixture_with_cauchy.ppf(q_cauchy_test)
        print("\nPPF values for Cauchy mixture:")
        for q, x_q in zip(q_cauchy_test, ppf_cauchy_results):
             if np.isfinite(x_q):
                  cdf_chk = mixture_with_cauchy.cdf(x_q)
                  print(f"  PPF({q:.3f}) = {x_q:9.4f} => CDF(x_q) = {cdf_chk:.4f} (Err: {cdf_chk-q:.1e})")
             else:
                  print(f"  PPF({q:.3f}) = {x_q:9.4f}")
        
        samples_cauchy_mix = mixture_with_cauchy.rvs(size=5000, random_state=123)
        plt.figure(figsize=(10,6))
        # For Cauchy, need to limit histogram range due to extreme outliers
        hist_range = (mixture_with_cauchy.ppf(0.01), mixture_with_cauchy.ppf(0.99))
        if not (np.isfinite(hist_range[0]) and np.isfinite(hist_range[1])): hist_range = (-20,20)

        plt.hist(samples_cauchy_mix, bins=100, density=True, range=hist_range, label='Cauchy Mix RVS (hist range limited)')
        x_plot_cauchy = np.linspace(hist_range[0], hist_range[1], 400)
        plt.plot(x_plot_cauchy, mixture_with_cauchy.pdf(x_plot_cauchy), 'r-', label='Cauchy Mix PDF')
        plt.title(f"Mixture with Cauchy Component\n{mixture_with_cauchy.name}")
        plt.ylim(0, max(mixture_with_cauchy.pdf(x_plot_cauchy)[np.isfinite(mixture_with_cauchy.pdf(x_plot_cauchy))])*1.1) # Adjust y-limit
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    except Exception as e:
        print(f"Error during Cauchy mixture test: {e}")

#endregion -----------------------------------------------------------------------------------------
#region Tests 1 for Importance Sampling (Toy, 1D)

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
x_max = 500

watershed_x_min = 150
watershed_x_max = 200

#%% Function to calculate y
def calc(x):
    # return np.where((x >= watershed_x_min) & (x <= watershed_x_max), (x**3+4**x), 0)
    # return np.where((x >= watershed_x_min) & (x <= watershed_x_max), norm(loc=7.5, scale=0.5).pdf(x), 0)
    # return np.where((x >= watershed_x_min) & (x <= watershed_x_max), norm(loc=7.5, scale=1.5).pdf(x), 0)
    # return np.where((x >= watershed_x_min) & (x <= watershed_x_max), norm(loc=175, scale=35).pdf(x), 0)
    return norm(loc=175, scale=35).pdf(x)

#%% Real expected value of y
y_exp_real = integrate.quad(calc, x_min, x_max)[0]/(x_max - x_min)
print (y_exp_real)

#%% Expected value of y using Monte Carlo simulation
dist_mc = uniform(loc=x_min, scale=x_max - x_min)
x_mc = dist_mc.rvs(100)
y_mc = calc(x_mc)
y_exp_mc = np.mean(y_mc)
print (y_exp_mc)

#%% Expected value of y using Importance Sampling simulation (TruncNorm)
param_std = 20 # at least 0.8 to be good
dist_is = truncnorm(loc=175, scale=param_std, a=(x_min-175)/param_std, b=(x_max-175)/param_std)
# dist_is = truncnorm(**truncnorm_params(175, param_std, x_min, x_max))
x_is = dist_is.rvs(100)
y_is = calc(x_is)
p = dist_mc.pdf(x_is) # PDF of original uniform distribution at sampled points
q = dist_is.pdf(x_is) # PDF of proposal truncated normal at sampled points
weights = np.where(q > 1e-9, p/q, 0)
y_exp_is = np.mean(y_is * weights)
# y_exp = np.mean(y)
print (y_exp_is)

#%% Expected value of y using Importance Sampling simulation (TruncGenNorm)
beta = 10 # at least ? to be good
dist_is = TruncatedGeneralizedNormal(
    loc=175,
    # scale=(watershed_x_max-watershed_x_min),
    scale=(200-150),
    lower_bound=x_min,
    upper_bound=x_max,
    beta=beta,
)
x_is = dist_is.rvs(10)
y_is = calc(x_is)
p = dist_mc.pdf(x_is) # PDF of original uniform distribution at sampled points
q = dist_is.pdf(x_is) # PDF of proposal truncated normal at sampled points
weights = np.where(q > 1e-9, p/q, 0)
y_exp_is = np.mean(y_is * weights)
# y_exp = np.mean(y)
print (y_exp_is)

#%% Probability of exceedence
y_val = 0.0001 #750_000
y_val = 0.003 #750_000

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
(pn.ggplot(pd.DataFrame(dict(x = y_mc)), pn.aes(x='x'))
    + pn.geom_histogram()
    + pn.scale_y_log10()
    + pn.labs(x = 'y (real values)')
)

# #%%
# (pn.ggplot(pd.DataFrame(dict(x = y_mc)).loc[lambda _: _.x > 0], pn.aes(x='x'))
#     + pn.geom_histogram()
#     + pn.scale_y_log10()
#     + pn.labs(x = 'y (real values, non-zero)')
# )

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
#region Tests 2 for Importance Sampling (Toy, 1D)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm

#%%
# 1. Define the true distribution p(x) and the function f(x)
# p(x) is Uniform(a, b)
a = 0.0
b = 10.0
p_dist = uniform(loc=a, scale=b - a) # scipy.stats.uniform

# f(x) is peaked at c, zero elsewhere
c = 2.0
f_peak_width = 0.5  # Controls how wide the non-zero part of f(x) is
f_active_half_width = 1.5 # f(x) will be non-zero in [c-f_active_half_width, c+f_active_half_width]

def f(x_arr):
    # Ensure x_arr is a numpy array for vectorized operations
    x_arr = np.asarray(x_arr)
    # Gaussian-like shape centered at c, scaled to have max 1
    # y = np.exp(-((x_arr - c)**2) / (2 * f_peak_width**2))
    # Make it exactly zero outside an "active region"
    # condition = (x_arr >= c - f_active_half_width) & (x_arr <= c + f_active_half_width)
    # return np.where(condition, y, 0.0)

    # Simpler f(x) for clarity: a triangular pulse
    # Tends to zero as x moves away from c
    # Max at c, zero at c ± active_half_width
    res = np.zeros_like(x_arr, dtype=float)
    # condition = np.abs(x_arr - c) <= f_active_half_width
    # res[condition] = 1.0 - np.abs(x_arr[condition] - c) / f_active_half_width

    # Using a Gaussian-like shape that's zero outside a range
    # This is more aligned with "tends to get smaller as x moves away from c"
    # and "mostly zero when significantly far away"
    y = np.exp(-((x_arr - c)**2) / (2 * f_peak_width**2))
    condition = (x_arr >= c - f_active_half_width) & (x_arr <= c + f_active_half_width)
    # A more practical f(x) could be like this:
    # if x is far from c, it's 0. If it's close, it has some value.
    return np.where(condition, y, 0.0)

#%%
# 2. Define the proposal distribution q(x) for Importance Sampling
# We'll use a truncated normal distribution centered at c, over [a,b]
q_mu = c
q_sigma = 0.75 #0.75 # Make it reasonably concentrated around c but not too narrow
# Parameters for truncnorm: (lower_clip - loc) / scale, (upper_clip - loc) / scale
clip_a, clip_b = (a - q_mu) / q_sigma, (b - q_mu) / q_sigma
q_dist = truncnorm(clip_a, clip_b, loc=q_mu, scale=q_sigma)

#%%
# Number of samples
n_samples = 500

#%%
# --- Standard Monte Carlo ---
print("--- Standard Monte Carlo ---")
x_mc = p_dist.rvs(size=n_samples)
y_mc = f(x_mc)

# Count non-zero samples for f(x)
non_zero_mc = np.sum(y_mc > 1e-9) # Use a small threshold for floating point
print(f"MC: Generated {n_samples} samples for x.")
print(f"MC: {non_zero_mc} samples resulted in f(x) > 0 (approx).")
print(f"MC: Proportion of useful samples: {non_zero_mc / n_samples:.4f}")

#%%
# --- Importance Sampling ---
print("\n--- Importance Sampling ---")
x_is = q_dist.rvs(size=n_samples) # Sample x from q(x)
y_is = f(x_is)

# Calculate importance weights: w(x) = p(x) / q(x)
# For p(x) = Uniform(a,b), p(x_i) = 1/(b-a) if a <= x_i <= b, else 0
# Since q_dist samples only from [a,b], all x_is are in this range.
p_val_at_x_is = p_dist.pdf(x_is) # This is 1/(b-a) for all x_is
q_val_at_x_is = q_dist.pdf(x_is)

# Defensive check for q_val_at_x_is being zero, though truncnorm should prevent this within its support
if np.any(q_val_at_x_is <= 1e-9): # Using a small threshold
    print("Warning: Some q(x_is) values are very small or zero!")
    # For those values, if p_val is non-zero, weight would be huge.
    # This indicates a poor choice of q(x) if p(x)f(x) is non-zero there.
    # In our setup, q_dist is defined over [a,b] and p_dist is uniform, so this shouldn't be an issue
    # unless f(x) is non-zero where q(x) is practically zero but p(x) is not.

weights = p_val_at_x_is / q_val_at_x_is
# Normalize weights (common practice, makes it robust to unnormalized q)
# For estimating expectations, E_p[g(X)] approx sum(w_i * g(x_i)) / sum(w_i)
# For histograms, many libraries handle weights directly. `density=True` with weights
# effectively does sum(w_i in bin) / (sum_total_weights * bin_width)
# If you sum weights to 1, then it becomes sum(normalized_w_i in bin) / bin_width

non_zero_is = np.sum(y_is > 1e-9)
print(f"IS: Generated {n_samples} samples for x (from q(x)).")
print(f"IS: {non_zero_is} samples resulted in f(x) > 0 (approx).")
print(f"IS: Proportion of 'active region' samples: {non_zero_is / n_samples:.4f}")
print(f"IS: Min weight: {np.min(weights):.4f}, Max weight: {np.max(weights):.4f}, Mean weight: {np.mean(weights):.4f}")
if np.var(weights) > 10 * np.mean(weights)**2 and np.var(weights) > 1: # Heuristic for high variance
    print(f"IS: Warning - High variance in weights detected (Var: {np.var(weights):.2f}). Consider a q(x) closer to p(x) or a mixture.")

#%%
# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: p(x), q(x) and f(x)
x_plot = np.linspace(a, b, 500)
f_x_plot = f(x_plot)
p_x_plot = p_dist.pdf(x_plot)
q_x_plot = q_dist.pdf(x_plot)

axs[0,0].plot(x_plot, p_x_plot, label=f'p(x) = Uniform({a},{b})', color='blue')
axs[0,0].plot(x_plot, q_x_plot, label=f'q(x) = TruncNorm(mu={q_mu}, sigma={q_sigma})', color='green', linestyle='--')
ax_f = axs[0,0].twinx() # Secondary y-axis for f(x)
ax_f.plot(x_plot, f_x_plot, label='f(x) (target function)', color='red', linestyle=':')
ax_f.set_ylabel('f(x) value', color='red')
ax_f.tick_params(axis='y', labelcolor='red')
ax_f.set_ylim(bottom=-0.05, top=np.max(f_x_plot)*1.1 if np.max(f_x_plot) > 0 else 1.1)


axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('Density (p(x), q(x))', color='blue')
axs[0,0].tick_params(axis='y', labelcolor='blue')
axs[0,0].set_title('Distributions p(x), q(x) and function f(x)')
lines, labels = axs[0,0].get_legend_handles_labels()
lines2, labels2 = ax_f.get_legend_handles_labels()
axs[0,0].legend(lines + lines2, labels + labels2, loc='upper right')
axs[0,0].grid(True, alpha=0.3)


# Plot 2: Histogram of x_mc samples and x_is samples
axs[0,1].hist(x_mc, bins=50, density=True, alpha=0.6, label='x_mc ~ p(x)', color='blue')
axs[0,1].hist(x_is, bins=50, density=True, alpha=0.6, label='x_is ~ q(x)', color='green') # density for x_is is q(x)
axs[0,1].plot(x_plot, p_x_plot, color='blue', linestyle='-', lw=1.5, label='True p(x)')
axs[0,1].plot(x_plot, q_x_plot, color='green', linestyle='--', lw=1.5, label='True q(x)')
axs[0,1].set_xlabel('x value')
axs[0,1].set_ylabel('Density of sampled x')
axs[0,1].set_title('Histograms of Sampled x values')
axs[0,1].legend()
axs[0,1].grid(True, alpha=0.3)


# Plot 3: Histograms of f(x) from MC
# We filter out the zeros for y_mc for a clearer plot of the non-zero part,
# but it's important to remember those zeros exist.
y_mc_non_zero = y_mc[y_mc > 1e-9]
if len(y_mc_non_zero) > 1: # Need at least 2 points for histogram
    axs[1,0].hist(y_mc_non_zero, bins=30, density=True, alpha=0.7, label=f'f(x_mc) (MC, {len(y_mc_non_zero)} non-zero)', color='blue')
else:
    axs[1,0].text(0.5, 0.5, 'MC: Too few non-zero f(x)\n to plot histogram', ha='center', va='center')
axs[1,0].set_xlabel('f(x) value')
axs[1,0].set_ylabel('Density')
axs[1,0].set_title('Distribution of f(x) from Standard MC')
axs[1,0].legend()
axs[1,0].grid(True, alpha=0.3)
axs[1,0].set_xlim(-0.05, 1.05) # f(x) values are between 0 and 1

# Plot 4: Histograms of f(x) from IS
# Filter out zeros for y_is for plotting clarity, but use corresponding weights
y_is_non_zero_indices = (y_is > 1e-9)
y_is_plot = y_is[y_is_non_zero_indices]
weights_plot = weights[y_is_non_zero_indices]

if len(y_is_plot) > 1:
    # For density=True, weights are used to make the integral of the histogram = 1.
    # The sum of weights is used in normalization.
    axs[1,1].hist(y_is_plot, bins=30, density=True, weights=weights_plot, alpha=0.7, label=f'f(x_is) (IS, {len(y_is_plot)} non-zero)', color='green')
else:
    axs[1,1].text(0.5, 0.5, 'IS: Too few non-zero f(x)\n to plot histogram', ha='center', va='center')
axs[1,1].set_xlabel('f(x) value')
axs[1,1].set_ylabel('Density')
axs[1,1].set_title('Distribution of f(x) from Importance Sampling')
axs[1,1].legend()
axs[1,1].grid(True, alpha=0.3)
axs[1,1].set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.show()

#%%
# Compare the distributions of f(x) (e.g. their means)
# Note: E_p[f(X)]
mean_f_mc = np.mean(y_mc)
# E_p[f(X)] = E_q[f(X) * p(X)/q(X)]
# Self-normalized IS estimator: sum(weights * f(x_is)) / sum(weights)
# If sum(weights) is used for normalization in histogram (density=True), mean from hist is what we want.
# For direct expectation:
if np.sum(weights) > 0: # Avoid division by zero if all weights are zero (unlikely)
    # mean_f_is = np.sum(weights * y_is) / np.sum(weights)
    mean_f_is = np.sum(weights * y_is) / n_samples
else:
    mean_f_is = 0 # Or handle as an error / NaN

print(f"\nMean of f(x) from MC: {mean_f_mc:.4f}")
print(f"Mean of f(x) from IS: {mean_f_is:.4f}")

# True mean of f(x) can be approximated by high-resolution numerical integration
# For a more precise 'true' distribution of f(x), one would need many more samples from p(x)
# or analytical derivation if f(x) is simple enough.
# Let's use a very large MC sample as a 'ground truth' reference for the PDF of f(x)
n_ref_samples = 500000
x_ref = p_dist.rvs(size=n_ref_samples)
y_ref = f(x_ref)

mean_f_ref = np.mean(y_ref)
print(f"Mean of f(x) from Ref: {mean_f_ref:.4f}")

y_ref_non_zero = y_ref[y_ref > 1e-9]

if len(y_ref_non_zero) > 1 and len(y_mc_non_zero) > 1:
    axs[1,0].hist(y_ref_non_zero, bins=30, density=True, alpha=0.4, label=f'Ref ({len(y_ref_non_zero)} non-zero)', color='grey', histtype='step', linewidth=1.5)
    axs[1,0].legend()
if len(y_ref_non_zero) > 1 and len(y_is_plot) > 1:
    axs[1,1].hist(y_ref_non_zero, bins=30, density=True, alpha=0.4, label=f'Ref ({len(y_ref_non_zero)} non-zero)', color='grey', histtype='step', linewidth=1.5)
    axs[1,1].legend()

plt.show()
# plt.savefig("importance_sampling_f_dist.png") # Save the figure again after adding ref
# print("\nPlots updated with reference distribution and saved to importance_sampling_f_dist.png")

#endregion -----------------------------------------------------------------------------------------
#region Tests 2b for Importance Sampling (Toy, 1D)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm
from statsmodels.distributions.empirical_distribution import ECDF # For weighted ECDF

# 1. Define the true distribution p(x) and the function f(x)
# p(x) is Uniform(a, b)
a = 0.0
b = 10.0
p_dist = uniform(loc=a, scale=b - a) # scipy.stats.uniform

# f(x) is peaked at c, zero elsewhere
c_val = 2.0 # Renamed from 'c' to avoid conflict with ECDF variable
f_peak_width = 0.5  # Controls how wide the non-zero part of f(x) is
f_active_half_width = 1.5 # f(x) will be non-zero in [c-f_active_half_width, c+f_active_half_width]

def f(x_arr):
    x_arr = np.asarray(x_arr)
    y = np.exp(-((x_arr - c_val)**2) / (2 * f_peak_width**2))
    condition = (x_arr >= c_val - f_active_half_width) & (x_arr <= c_val + f_active_half_width)
    return np.where(condition, y, 0.0)

# 2. Define the proposal distribution q(x) for Importance Sampling
q_mu = c_val
q_sigma = 0.75 # Original value, you might experiment with this
clip_a, clip_b = (a - q_mu) / q_sigma, (b - q_mu) / q_sigma
q_dist = truncnorm(clip_a, clip_b, loc=q_mu, scale=q_sigma)

# Number of samples
n_samples = 2000 # Same as before
n_ref_samples = 500000 # For ground truth

# --- Standard Monte Carlo ---
print("--- Standard Monte Carlo ---")
x_mc = p_dist.rvs(size=n_samples)
y_mc = f(x_mc)
non_zero_mc = np.sum(y_mc > 1e-9)
print(f"MC: {non_zero_mc}/{n_samples} non-zero f(x) samples.")

# --- Importance Sampling ---
print("\n--- Importance Sampling ---")
x_is = q_dist.rvs(size=n_samples)
y_is = f(x_is)
p_val_at_x_is = p_dist.pdf(x_is)
q_val_at_x_is = q_dist.pdf(x_is)
weights = p_val_at_x_is / q_val_at_x_is
non_zero_is = np.sum(y_is > 1e-9)
print(f"IS: {non_zero_is}/{n_samples} non-zero f(x) samples (before weighting).")
print(f"IS: Mean weight: {np.mean(weights):.4f}, Var weight: {np.var(weights):.4f}")
# For self-normalized estimates, we need the sum of weights
sum_weights = np.sum(weights)
normalized_weights_for_ecdf = weights / sum_weights # statsmodels ECDF expects weights summing to 1 if side='right' (default for probability)


# --- Ground Truth (Large MC Sample) ---
print("\n--- Ground Truth Generation ---")
x_ref = p_dist.rvs(size=n_ref_samples)
y_ref = f(x_ref)
non_zero_ref = np.sum(y_ref > 1e-9)
print(f"REF: {non_zero_ref}/{n_ref_samples} non-zero f(x) samples.")

# --- Calculate ECDFs ---

# Standard MC ECDF
# Sort the samples. The ECDF jumps by 1/n at each sample.
# For plotting, we want to include the point mass at y=0 correctly if present.
# We can use statsmodels ECDF for unweighted case too for consistency, or do it manually.
# y_mc_sorted = np.sort(y_mc)
# ecdf_mc_y = np.arange(1, n_samples + 1) / n_samples
# To handle the 0s better visually for ECDF:
# We can plot from slightly before min(y_mc) to slightly after max(y_mc)
ecdf_mc = ECDF(y_mc)

# Importance Sampling Weighted ECDF
# For a weighted ECDF, we sort the y_is values.
# Then, at each y_is_sorted[j], the ECDF jumps by its normalized weight w_sorted[j].
# statsmodels.distributions.empirical_distribution.ECDF can handle weights.
# It expects weights to sum to 1 for the typical interpretation, or it normalizes internally.
# Let's use the normalized weights explicitly to be clear.
# Sort y_is and align weights:
sort_indices_is = np.argsort(y_is)
y_is_sorted = y_is[sort_indices_is]
weights_sorted_for_ecdf = normalized_weights_for_ecdf[sort_indices_is] # Use weights that sum to 1
ecdf_is = ECDF(y_is_sorted, weights=weights_sorted_for_ecdf) # weights here should be probability weights

# Ground Truth ECDF
ecdf_ref = ECDF(y_ref)

# --- Plotting ECDFs ---
plt.figure(figsize=(10, 7))

# Generate points for plotting the ECDF steps
# ECDF objects from statsmodels can be called like functions
plot_y_vals = np.linspace(min(y_ref.min(), y_mc.min(), y_is.min()) - 0.01,
                          max(y_ref.max(), y_mc.max(), y_is.max()) + 0.01,
                          1000)
plot_y_vals = np.unique(np.concatenate([plot_y_vals, y_mc, y_is, y_ref])) # Add exact sample points
plot_y_vals.sort()
plot_y_vals = np.clip(plot_y_vals, -0.05, 1.05) # f(x) is between 0 and 1

# Plot MC ECDF
plt.plot(ecdf_mc.x, ecdf_mc.y, label=f'Monte Carlo (n={n_samples})', drawstyle='steps-post', color='blue', alpha=0.7)

# Plot IS ECDF
# The ECDF object handles the steps correctly.
plt.plot(ecdf_is.x, ecdf_is.y, label=f'Importance Sampling (n={n_samples})', drawstyle='steps-post', color='green', alpha=0.7)

# Plot Reference ECDF
plt.plot(ecdf_ref.x, ecdf_ref.y, label=f'Ground Truth (n={n_ref_samples})', drawstyle='steps-post', color='grey', linestyle='--', linewidth=1.5)


plt.xlabel('f(x) value (y)')
plt.ylabel('Cumulative Probability P(f(X) <= y)')
plt.title('Empirical Cumulative Distribution Functions (ECDF) of f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05) # f(x) values are mostly between 0 and 1
plt.ylim(-0.05, 1.05)
plt.show()

# --- For your reference, how to calculate mean from ECDF (conceptually) ---
# Mean from ECDF = integral of (1 - CDF(y)) dy from 0 to inf (for non-negative variables)
# - integral of CDF(y) dy from -inf to 0
# Or for discrete values from ECDF object:
# mean_mc_from_ecdf = np.sum(np.diff(ecdf_mc.x) * (1 - ecdf_mc.y[:-1])) # If ecdf_mc.x starts at min value
# This is more complex for step functions, direct mean calculation is easier.

# Direct mean calculations (as before, for context):
mean_f_mc = np.mean(y_mc)
if sum_weights > 0:
    mean_f_is_self_norm = np.sum(weights * y_is) / sum_weights
    mean_f_is_basic = np.sum(weights * y_is) / n_samples
else:
    mean_f_is_self_norm = mean_f_is_basic = 0
mean_f_ref = np.mean(y_ref)

print(f"\nMean of f(x) from MC: {mean_f_mc:.6f}")
print(f"Mean of f(x) from IS (Self-Norm): {mean_f_is_self_norm:.6f}")
print(f"Mean of f(x) from IS (Basic): {mean_f_is_basic:.6f}")
print(f"Mean of f(x) from Ref (Ground Truth): {mean_f_ref:.6f}")

#endregion -----------------------------------------------------------------------------------------
#region Tests 2c for Importance Sampling (Toy, 1D)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm
from statsmodels.distributions.empirical_distribution import ECDF # For unweighted

# --- (Keep your existing p(x), f(x), q(x) definitions and sampling code) ---
# 1. Define the true distribution p(x) and the function f(x)
# p(x) is Uniform(a, b)
a = 0.0
b = 10.0
p_dist = uniform(loc=a, scale=b - a) # scipy.stats.uniform

# f(x) is peaked at c, zero elsewhere
c_val = 2.0 # Renamed from 'c' to avoid conflict with ECDF variable
f_peak_width = 0.5  # Controls how wide the non-zero part of f(x) is
f_active_half_width = 1.5 # f(x) will be non-zero in [c-f_active_half_width, c+f_active_half_width]

def f(x_arr):
    x_arr = np.asarray(x_arr)
    y = np.exp(-((x_arr - c_val)**2) / (2 * f_peak_width**2))
    condition = (x_arr >= c_val - f_active_half_width) & (x_arr <= c_val + f_active_half_width)
    return np.where(condition, y, 0.0)

# 2. Define the proposal distribution q(x) for Importance Sampling
q_mu = c_val
q_sigma = 0.75 # Original value, you might experiment with this
clip_a, clip_b = (a - q_mu) / q_sigma, (b - q_mu) / q_sigma
q_dist = truncnorm(clip_a, clip_b, loc=q_mu, scale=q_sigma)

# Number of samples
n_samples = 2000 # Same as before
n_ref_samples = 500000 # For ground truth

# --- Standard Monte Carlo ---
print("--- Standard Monte Carlo ---")
x_mc = p_dist.rvs(size=n_samples)
y_mc = f(x_mc)
non_zero_mc = np.sum(y_mc > 1e-9)
print(f"MC: {non_zero_mc}/{n_samples} non-zero f(x) samples.")

# --- Importance Sampling ---
print("\n--- Importance Sampling ---")
x_is = q_dist.rvs(size=n_samples)
y_is = f(x_is)
p_val_at_x_is = p_dist.pdf(x_is)
q_val_at_x_is = q_dist.pdf(x_is)
weights = p_val_at_x_is / q_val_at_x_is
non_zero_is = np.sum(y_is > 1e-9)
print(f"IS: {non_zero_is}/{n_samples} non-zero f(x) samples (before weighting).")
print(f"IS: Mean weight: {np.mean(weights):.4f}, Var weight: {np.var(weights):.4f}")
sum_weights = np.sum(weights)
if sum_weights == 0: # Avoid division by zero if all weights are zero
    normalized_weights_is = np.ones_like(weights) / len(weights) # Fallback to uniform if sum is 0
else:
    normalized_weights_is = weights / sum_weights


# --- Ground Truth (Large MC Sample) ---
print("\n--- Ground Truth Generation ---")
x_ref = p_dist.rvs(size=n_ref_samples)
y_ref = f(x_ref)
non_zero_ref = np.sum(y_ref > 1e-9)
print(f"REF: {non_zero_ref}/{n_ref_samples} non-zero f(x) samples.")

# --- Calculate ECDFs ---

# Standard MC ECDF (using statsmodels ECDF for unweighted)
ecdf_mc = ECDF(y_mc)

# Ground Truth ECDF (using statsmodels ECDF for unweighted)
ecdf_ref = ECDF(y_ref)

# Importance Sampling Weighted ECDF (Manual Calculation)
# 1. Sort y_is and keep weights aligned
idx_sort_is = np.argsort(y_is)
y_is_sorted = y_is[idx_sort_is]
weights_is_sorted = normalized_weights_is[idx_sort_is] # Use weights that sum to 1

# 2. Calculate cumulative sum of weights
ecdf_is_y_values = np.cumsum(weights_is_sorted)

# 3. Define the x-values for the ECDF steps
# The ECDF value ecdf_is_y_values[i] is valid for y_is_sorted[i] <= y < y_is_sorted[i+1]
# To plot with steps-post, we use y_is_sorted as the x-coordinates where the step *up* occurs.
# We also need to ensure the plot starts at 0 if the smallest y_is_sorted is > 0 and has weight at y=0
# And handle the case where all y_is are 0.

# For plotting, we need to make sure the ECDF starts appropriately.
# If the smallest y_is_sorted > 0, the CDF is 0 before that point.
# If y_is_sorted starts with 0s, their cumulative weight will be the P(Y=0).

# Prepend a point to make the ECDF start from 0 if necessary,
# and to correctly plot the first step.
# `ecdf_is_x_plot` will be the "x" coordinates of the ECDF steps.
# `ecdf_is_y_plot` will be the "y" coordinates (cumulative probabilities).

# Handle empty y_is_sorted (e.g., if n_samples = 0 or f(x) always 0 and no weights)
if len(y_is_sorted) == 0:
    ecdf_is_x_plot = np.array([-np.inf, np.inf]) # Or some reasonable range like [0,1]
    ecdf_is_y_plot = np.array([0,0]) # No probability mass
elif np.all(y_is_sorted == y_is_sorted[0]): # All y_is_sorted values are the same
    # e.g. all f(x) are 0.
    ecdf_is_x_plot = np.array([y_is_sorted[0], y_is_sorted[0]])
    # If they are all the same non-zero value, y_plot should be [0,1] for that value.
    # If y_is_sorted[0] > 0, need to show 0 before it.
    if y_is_sorted[0] > 0:
        ecdf_is_x_plot = np.array([y_is_sorted[0] - 1e-9, y_is_sorted[0], y_is_sorted[0]]) # Approximate for plotting
        ecdf_is_y_plot = np.array([0, 0, 1.0])
    else: # all y_is_sorted are 0
        ecdf_is_x_plot = np.array([0, 0])
        ecdf_is_y_plot = np.array([0, 1.0]) # CDF is 1 at y=0
else:
    # Standard case
    ecdf_is_x_plot = y_is_sorted
    # The y_values are the cumulative sums. For plotting with steps-post,
    # the value ecdf_is_y_values[i] is achieved *at and after* ecdf_is_x_plot[i].
    # We need to prepend a 0 to the y-values if the first x-value is > 0,
    # and adjust the x-values to match for steps-post.
    
    # More robust way using unique values and their cumulative probabilities
    unique_y_is, unique_idx = np.unique(y_is_sorted, return_index=True)
    # Sum weights for each unique y value
    # This is a bit more complex if there are repeated y values with different original indices
    # Easier: Iterate through sorted unique y values
    
    # Let's use a simpler approach suitable for plotting with steps-post:
    # `ecdf_is_x_plot` will be unique sorted y values.
    # `ecdf_is_y_plot_final` will be the CDF value *at* these x values.
    
    # This is essentially what statsmodels.ECDF does internally for the .x and .y attributes
    # Find unique sorted values for x-axis
    unique_x = np.unique(y_is_sorted)
    
    # Calculate CDF values at these unique points
    # For each u in unique_x, CDF(u) = sum of weights for all y_is <= u
    cdf_at_unique_x = np.array([np.sum(weights_is_sorted[y_is_sorted <= u_val]) for u_val in unique_x])

    # For plotting with steps-post, we need to handle the start properly.
    # If the smallest unique_x > 0, the CDF is 0 before that.
    ecdf_is_x_plot_final = unique_x
    ecdf_is_y_plot_final = cdf_at_unique_x

    # Ensure plot starts at y=0 if smallest x is > 0
    if ecdf_is_x_plot_final[0] > 0:
        ecdf_is_x_plot_for_plot = np.insert(ecdf_is_x_plot_final, 0, ecdf_is_x_plot_final[0]-1e-9) # A point just before
        ecdf_is_y_plot_for_plot = np.insert(ecdf_is_y_plot_final, 0, 0)
        # And also add the first actual point again to make the step up from 0
        ecdf_is_x_plot_for_plot = np.insert(ecdf_is_x_plot_for_plot, 1, ecdf_is_x_plot_final[0])
        ecdf_is_y_plot_for_plot = np.insert(ecdf_is_y_plot_for_plot, 1, 0)

    else: # Smallest x is 0
        # If first x is 0, cdf_at_unique_x[0] is P(Y=0).
        # For steps-post, we want the value *before* the jump at 0 to be 0.
        ecdf_is_x_plot_for_plot = np.insert(ecdf_is_x_plot_final, 0, -1e-9) # A point just before 0
        ecdf_is_y_plot_for_plot = np.insert(ecdf_is_y_plot_final, 0, 0)
        
    # Ensure the plot goes to 1 at the end
    if ecdf_is_y_plot_for_plot[-1] < 1.0 - 1e-6 : # If not already very close to 1
        ecdf_is_x_plot_for_plot = np.append(ecdf_is_x_plot_for_plot, ecdf_is_x_plot_for_plot[-1] + 1e-9) # A point just after
        ecdf_is_y_plot_for_plot = np.append(ecdf_is_y_plot_for_plot, 1.0)


# --- Plotting ECDFs ---
plt.figure(figsize=(10, 7))

# Plot MC ECDF
plt.plot(ecdf_mc.x, ecdf_mc.y, label=f'Monte Carlo (n={n_samples})', drawstyle='steps-post', color='blue', alpha=0.7)

# Plot IS ECDF (Manual)
if len(y_is_sorted) > 0: # Only plot if there are samples
    plt.plot(ecdf_is_x_plot_for_plot, ecdf_is_y_plot_for_plot, label=f'Importance Sampling (n={n_samples})', drawstyle='steps-post', color='green', alpha=0.7)
else:
    plt.plot([], [], label=f'Importance Sampling (n={n_samples}) - No Data', drawstyle='steps-post', color='green', alpha=0.7)


# Plot Reference ECDF
plt.plot(ecdf_ref.x, ecdf_ref.y, label=f'Ground Truth (n={n_ref_samples})', drawstyle='steps-post', color='grey', linestyle='--', linewidth=1.5)

plt.xlabel('f(x) value (y)')
plt.ylabel('Cumulative Probability P(f(X) <= y)')
plt.title('Empirical Cumulative Distribution Functions (ECDF) of f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()

# --- (Mean calculations remain the same) ---
mean_f_mc = np.mean(y_mc)
if sum_weights > 0:
    mean_f_is_self_norm = np.sum(weights * y_is) / sum_weights
    mean_f_is_basic = np.sum(weights * y_is) / n_samples
else:
    mean_f_is_self_norm = mean_f_is_basic = 0
mean_f_ref = np.mean(y_ref)

print(f"\nMean of f(x) from MC: {mean_f_mc:.6f}")
print(f"Mean of f(x) from IS (Self-Norm): {mean_f_is_self_norm:.6f}")
print(f"Mean of f(x) from IS (Basic): {mean_f_is_basic:.6f}")
print(f"Mean of f(x) from Ref (Ground Truth): {mean_f_ref:.6f}")

#endregion -----------------------------------------------------------------------------------------
#region Tests 2d for Importance Sampling (Toy, 1D, mixture model)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm
from statsmodels.distributions.empirical_distribution import ECDF

# 1. Define the true distribution p(x) and the function f(x)
a = 0.0
b = 10.0
p_dist = uniform(loc=a, scale=b - a)

c_val = 2.0
f_peak_width = 0.5
f_active_half_width = 1.5

def f(x_arr):
    x_arr = np.asarray(x_arr)
    y = np.exp(-((x_arr - c_val)**2) / (2 * f_peak_width**2))
    condition = (x_arr >= c_val - f_active_half_width) & (x_arr <= c_val + f_active_half_width)
    return np.where(condition, y, 0.0)

# 2. Define the PEAK component of the proposal distribution q_peak(x)
q_peak_mu = c_val
q_peak_sigma = 0.75 # Original sigma for the peaked component
clip_a_peak, clip_b_peak = (a - q_peak_mu) / q_peak_sigma, (b - q_peak_mu) / q_peak_sigma
q_peak_dist = truncnorm(clip_a_peak, clip_b_peak, loc=q_peak_mu, scale=q_peak_sigma)

# Number of samples
n_samples = 2000
n_ref_samples = 500000

# --- Standard Monte Carlo ---
print("--- Standard Monte Carlo ---")
x_mc = p_dist.rvs(size=n_samples)
y_mc = f(x_mc)
ecdf_mc = ECDF(y_mc)
mean_f_mc = np.mean(y_mc)
print(f"MC: Mean f(x) = {mean_f_mc:.6f}")

# --- Importance Sampling (Original - for comparison if you uncomment) ---
# print("\n--- Importance Sampling (Original) ---")
# x_is_orig = q_peak_dist.rvs(size=n_samples)
# y_is_orig = f(x_is_orig)
# p_val_at_x_is_orig = p_dist.pdf(x_is_orig)
# q_val_at_x_is_orig = q_peak_dist.pdf(x_is_orig) # Using q_peak as the original q
# weights_orig = p_val_at_x_is_orig / q_val_at_x_is_orig
# sum_weights_orig = np.sum(weights_orig)
# mean_f_is_orig_self_norm = np.sum(weights_orig * y_is_orig) / sum_weights_orig if sum_weights_orig > 0 else 0
# print(f"IS (Orig): Mean weight: {np.mean(weights_orig):.4f}, Var weight: {np.var(weights_orig):.4f}")
# print(f"IS (Orig): Self-Norm Mean f(x) = {mean_f_is_orig_self_norm:.6f}")
# Manual ECDF for IS Original (copying logic from previous solution)
# idx_sort_is_orig = np.argsort(y_is_orig)
# y_is_orig_sorted = y_is_orig[idx_sort_is_orig]
# normalized_weights_is_orig = weights_orig / sum_weights_orig if sum_weights_orig > 0 else np.ones_like(weights_orig)/len(weights_orig)
# weights_is_orig_sorted = normalized_weights_is_orig[idx_sort_is_orig]
# unique_x_orig, cdf_at_unique_x_orig, x_plot_orig, y_plot_orig = None, None, None, None
# if len(y_is_orig_sorted) > 0:
#     unique_x_orig = np.unique(y_is_orig_sorted)
#     cdf_at_unique_x_orig = np.array([np.sum(weights_is_orig_sorted[y_is_orig_sorted <= u_val]) for u_val in unique_x_orig])
#     x_plot_orig, y_plot_orig = manual_ecdf_plotting_points(unique_x_orig, cdf_at_unique_x_orig)


# --- Importance Sampling with Mixture Distribution ---
print("\n--- Importance Sampling (Mixture) ---")
alpha_mix = 0.2  # Mixture coefficient for p(x)

# Sampling from the mixture:
# Decide for each sample whether it comes from p(x) or q_peak(x)
# choices = np.random.choice([0, 1], size=n_samples, p=[alpha_mix, 1 - alpha_mix])
# n_from_p = np.sum(choices == 0)
# n_from_q_peak = n_samples - n_from_p
# More direct way to get counts:
n_from_p = np.random.binomial(n_samples, alpha_mix)
n_from_q_peak = n_samples - n_from_p

x_is_mix = np.zeros(n_samples)
if n_from_p > 0:
    x_is_mix[:n_from_p] = p_dist.rvs(size=n_from_p)
if n_from_q_peak > 0:
    x_is_mix[n_from_p:] = q_peak_dist.rvs(size=n_from_q_peak)
np.random.shuffle(x_is_mix) # Shuffle to mix them up, though not strictly necessary for math

y_is_mix = f(x_is_mix)

# PDF of the mixture distribution q_mix(x)
pdf_p_at_x_is_mix = p_dist.pdf(x_is_mix)
pdf_q_peak_at_x_is_mix = q_peak_dist.pdf(x_is_mix)
pdf_q_mix_at_x_is_mix = alpha_mix * pdf_p_at_x_is_mix + (1 - alpha_mix) * pdf_q_peak_at_x_is_mix

# Importance weights for the mixture
weights_mix = pdf_p_at_x_is_mix / pdf_q_mix_at_x_is_mix

# Diagnostics for mixture weights
print(f"IS (Mix) with alpha={alpha_mix}:")
print(f"  Mean weight: {np.mean(weights_mix):.4f}")
print(f"  Var weight:  {np.var(weights_mix):.4f}")
sum_weights_mix = np.sum(weights_mix)

# Mean estimates for mixture IS
mean_f_is_mix_self_norm = 0
mean_f_is_mix_basic = 0
if sum_weights_mix > 0:
    mean_f_is_mix_self_norm = np.sum(weights_mix * y_is_mix) / sum_weights_mix
if n_samples > 0 : # Basic estimator needs n_samples
    mean_f_is_mix_basic = np.sum(weights_mix * y_is_mix) / n_samples

print(f"  Self-Norm Mean f(x): {mean_f_is_mix_self_norm:.6f}")
print(f"  Basic Mean f(x):     {mean_f_is_mix_basic:.6f}")


# --- Ground Truth (Large MC Sample) ---
print("\n--- Ground Truth Generation ---")
x_ref = p_dist.rvs(size=n_ref_samples)
y_ref = f(x_ref)
ecdf_ref = ECDF(y_ref)
mean_f_ref = np.mean(y_ref)
print(f"REF: Mean f(x) = {mean_f_ref:.6f}")


# --- Manual ECDF Calculation Function (from previous solution) ---
def manual_ecdf_plotting_points(y_sorted_data, normalized_weights_sorted_data):
    if len(y_sorted_data) == 0:
        return np.array([-0.05, 1.05]), np.array([0, 0]) # Default for empty

    unique_x = np.unique(y_sorted_data)
    # This part needs care: sum weights for unique values
    # Easier: use y_sorted_data and cumsum of its corresponding normalized_weights_sorted_data
    # if all weights are simple 1/N, this is easy. For general weights:
    
    # This approach assumes y_sorted_data already has unique values or we sum weights for them.
    # Let's use the logic for unique points and their cumulative weights directly from statsmodels-like .x, .y
    
    # Rebuild ECDF .x and .y attributes similar to statsmodels
    # 1. Sort data and weights together
    idx_sort = np.argsort(y_sorted_data)
    y_s = y_sorted_data[idx_sort]
    w_s = normalized_weights_sorted_data[idx_sort]

    # 2. Find unique sorted y values
    unique_y_vals, unique_indices = np.unique(y_s, return_index=True)
    
    # 3. Calculate cumulative sum of weights. Need to sum weights for ties in y_s.
    # This is simpler: Use y_s and w_s directly. The ECDF points are (y_s[i], cumsum(w_s)[i])
    # For plotting steps-post, this already gives the "y" values of the ECDF.
    # The "x" values are y_s.
    
    cdf_y_values = np.cumsum(w_s)
    
    # For plotting:
    # Start from (y_s[0], cdf_y_values[0])
    # Before y_s[0], the CDF is 0, if y_s[0] > 0 or if y_s[0]=0 and it's the first point.
    
    plot_x = y_s
    plot_y = cdf_y_values

    # Prepend points to make the ECDF start from (y_min, 0) and correctly show first step.
    # Also ensure it ends at 1.0
    final_x = []
    final_y = []

    # Start from 0 probability
    if plot_x[0] > 0 : # If smallest value is > 0, CDF is 0 before it
        final_x.extend([plot_x[0] - 1e-9, plot_x[0]]) # Point just before, and the point itself
        final_y.extend([0, 0]) 
    elif plot_x[0] == 0: # If smallest value is 0
        final_x.append(-1e-9) # Point just before 0
        final_y.append(0)
    # else smallest value is < 0, which shouldn't happen for f(x) >=0

    final_x.extend(plot_x)
    final_y.extend(plot_y)
    
    # Ensure ECDF ends at 1.0
    if final_y[-1] < 1.0 - 1e-6:
        final_x.append(final_x[-1] + 1e-9) # Point just after last data point
        final_y.append(1.0) # Force to 1
    else: # If already 1.0 or very close, ensure the last y is exactly 1.0
        final_y[-1] = 1.0
        
    # Remove duplicates in x that might have arisen from prepending, keeping last y
    # This can be complex. The `statsmodels.ECDF` object handles this well with its .x and .y
    # For manual plotting, we can rely on `drawstyle='steps-post'` with sorted unique x and corresponding cdf y.
    
    # Let's simplify: use unique x values and calculate CDF at these points
    # This matches the logic from the previous working manual ECDF.
    
    unique_x_vals = np.unique(y_sorted_data)
    cdf_at_unique_x = np.array([np.sum(normalized_weights_sorted_data[y_sorted_data <= u_val]) for u_val in unique_x_vals])
    
    plot_x_final = unique_x_vals
    plot_y_final = cdf_at_unique_x
    
    if len(plot_x_final) == 0: return np.array([-0.05,1.05]), np.array([0,0])

    if plot_x_final[0] > 0:
        plot_x_for_plt = np.insert(plot_x_final, 0, plot_x_final[0] - 1e-9)
        plot_y_for_plt = np.insert(plot_y_final, 0, 0.0)
        plot_x_for_plt = np.insert(plot_x_for_plt, 1, plot_x_final[0]) # step up point
        plot_y_for_plt = np.insert(plot_y_for_plt, 1, 0.0)
    else: # plot_x_final[0] == 0
        plot_x_for_plt = np.insert(plot_x_final, 0, -1e-9)
        plot_y_for_plt = np.insert(plot_y_final, 0, 0.0)
        
    # Ensure last point goes to 1.0
    if plot_y_for_plt[-1] < 1.0 - 1e-6:
        plot_x_for_plt = np.append(plot_x_for_plt, plot_x_for_plt[-1] + 1e-9) # add a point slightly after
        plot_y_for_plt = np.append(plot_y_for_plt, 1.0) # force CDF to 1
    else:
        plot_y_for_plt[-1] = 1.0 # Ensure it's exactly 1

    return plot_x_for_plt, plot_y_for_plt


# ECDF for IS Mixture (Manual)
ecdf_is_mix_x_plot, ecdf_is_mix_y_plot = [],[]
if n_samples > 0 and sum_weights_mix > 0:
    normalized_weights_is_mix = weights_mix / sum_weights_mix
    ecdf_is_mix_x_plot, ecdf_is_mix_y_plot = manual_ecdf_plotting_points(y_is_mix, normalized_weights_is_mix)
else: # Handle case with no samples or zero sum of weights
    ecdf_is_mix_x_plot, ecdf_is_mix_y_plot = manual_ecdf_plotting_points(np.array([]), np.array([]))


# --- Plotting ECDFs ---
plt.figure(figsize=(12, 8))

# Plot MC ECDF
plt.plot(ecdf_mc.x, ecdf_mc.y, label=f'Monte Carlo (n={n_samples})', drawstyle='steps-post', color='blue', alpha=0.6, linewidth=1.5)

# Plot IS Mixture ECDF
if len(ecdf_is_mix_x_plot) > 0:
    plt.plot(ecdf_is_mix_x_plot, ecdf_is_mix_y_plot, label=f'IS Mixture alpha={alpha_mix} (n={n_samples})', drawstyle='steps-post', color='red', alpha=0.7, linewidth=1.5)
else:
    plt.plot([],[], label=f'IS Mixture alpha={alpha_mix} (n={n_samples}) - No Data', drawstyle='steps-post', color='red')

# Plot Reference ECDF
plt.plot(ecdf_ref.x, ecdf_ref.y, label=f'Ground Truth (n={n_ref_samples})', drawstyle='steps-post', color='grey', linestyle='--', linewidth=2)

plt.xlabel('f(x) value (y)')
plt.ylabel('Cumulative Probability P(f(X) <= y)')
plt.title('Empirical Cumulative Distribution Functions (ECDF) of f(x)')
plt.legend(loc='center right')
plt.grid(True, alpha=0.4)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()

# --- Print final mean comparisons ---
print(f"\n--- Final Mean Comparisons ---")
print(f"MC:              {mean_f_mc:.6f}")
# print(f"IS (Original SN): {mean_f_is_orig_self_norm:.6f}") # If you run original IS
print(f"IS (Mix Basic):    {mean_f_is_mix_basic:.6f}")
print(f"IS (Mix SelfNorm): {mean_f_is_mix_self_norm:.6f}")
print(f"Ground Truth:      {mean_f_ref:.6f}")

# Further diagnostics: Effective Sample Size for IS Mixture
if np.mean(weights_mix)**2 > 0: # Avoid division by zero if mean weight is zero
    ess_mix = n_samples / (1 + np.var(weights_mix) / (np.mean(weights_mix)**2) )
    print(f"IS (Mix) Effective Sample Size (ESS): {ess_mix:.2f} (out of {n_samples})")
else:
    print("IS (Mix) Effective Sample Size (ESS): N/A (mean weight is zero)")

#endregion -----------------------------------------------------------------------------------------
#region Tests 3 for Importance Sampling (Toy, 1D)

#%%
import numpy as np
from scipy.stats import norm

# --- Define the problem ---
# 1. Target distribution p(x)
p_mean = 2.0
p_std = 1.0
p_dist = norm(loc=p_mean, scale=p_std)

# 2. Proposal distribution q(x)
q_mean = 0.0
q_std = 2.0
q_dist = norm(loc=q_mean, scale=q_std)

# 3. Function of interest f(x)
def f(x):
    return x**2

# --- Importance Sampling Parameters ---
N = 10000  # Number of samples

# --- Generate Samples from Proposal q(x) ---
# X_i ~ q(x)
samples_X = q_dist.rvs(size=N)

# --- Calculate f(X_i) ---
fx_values = f(samples_X)

# --- Calculate Importance Weights w(X_i) = p(X_i) / q(X_i) ---
# In this example, p and q are fully known and normalized,
# so these are the 'true' weights w_i.
# If p_tilde was unnormalized, these would be w_tilde_i.
p_pdf_values = p_dist.pdf(samples_X)
q_pdf_values = q_dist.pdf(samples_X)

# Avoid division by zero if a sample falls in an extremely unlikely region for q
# (though for Gaussians this is less of an issue unless samples are astronomical)
q_pdf_values[q_pdf_values == 0] = 1e-100 # A very small number to prevent NaN/Inf
weights = p_pdf_values / q_pdf_values

# --- 1. Basic (Unnormalized) Importance Sampling Estimator ---
# This is appropriate because our p_dist.pdf and q_dist.pdf are normalized.
# mu_hat_IS = (1/N) * Σ [f(X_i) * w(X_i)]
terms_is = fx_values * weights
mu_hat_is = np.mean(terms_is)

# Standard Error for mu_hat_IS
# SE_hat(mu_hat_IS) = sqrt[ (1/(N(N-1))) * Σ (f(X_i)w(X_i) - mu_hat_IS)^2 ]
# Or, equivalently, std(terms_is) / sqrt(N)
# np.var(terms_is, ddof=1) gives the sample variance: (1/(N-1)) * Σ (term_i - mean(terms))^2
variance_of_terms_is = np.var(terms_is, ddof=1) # Sample variance of Y_i = f(X_i)w(X_i)
var_mu_hat_is = variance_of_terms_is / N          # Variance of the mean
se_mu_hat_is = np.sqrt(var_mu_hat_is)

# --- 2. Self-Normalized (Weighted) Importance Sampling Estimator ---
# mu_hat_SNIS = [ Σ f(X_i) * w_tilde(X_i) ] / [ Σ w_tilde(X_i) ]
# Here, w_tilde(X_i) is the same as w(X_i) because p and q were normalized.
# If p_dist.pdf was unnormalized p_tilde, then 'weights' would be w_tilde.
sum_weighted_fx = np.sum(fx_values * weights)
sum_weights = np.sum(weights)

if sum_weights == 0: # Should not happen with proper q and non-zero N
    mu_hat_snis = np.nan
    se_mu_hat_snis = np.nan
else:
    mu_hat_snis = sum_weighted_fx / sum_weights

    # Standard Error for mu_hat_SNIS
    # A common formula (derived from delta method or ratio estimator theory):
    # Var_hat(μ_hat_SNIS) ≈ [ N / (N-1) ] * [ Σ (w_i * (f(X_i) - μ_hat_SNIS))^2 ] / (Σ w_j)^2
    # For large N, N/(N-1) ≈ 1.
    # Let's use the formula with N/(N-1) for better small-N accuracy.
    
    # Numerator terms for variance calculation
    var_terms_snis = (weights * (fx_values - mu_hat_snis))**2
    numerator_var_snis = np.sum(var_terms_snis)
    denominator_var_snis = sum_weights**2

    if N > 1 and denominator_var_snis > 0:
        # The N/(N-1) factor here is to make the variance estimate unbiased for the *linearized* variable.
        # Some texts present it as (1/ (sum_weights^2)) * sum (w_i * (f(X_i) - mu_hat_SNIS))^2 * (1/(N-1)) * N
        # which simplifies to the N/(N-1) factor or simply using N in the sum if N is large.
        # A more direct variance for a weighted mean (often seen):
        # sum_sq_weighted_deviations = np.sum( (weights * (fx_values - mu_hat_snis))**2 )
        # var_mu_hat_snis_approx = sum_sq_weighted_deviations / (sum_weights**2)
        # However, the formula below is generally preferred for ratio estimators.

        # Let's use the formula that often appears, which is equivalent to
        # var_hat(mu_hat_snis) = (1/N) * (1/(N-1)) * sum_i (N * w_i_norm * (f(x_i) - mu_hat_snis) )^2
        # where w_i_norm = w_i / sum(w_j). More directly from books:
        
        var_mu_hat_snis = (N / (N - 1.0)) * numerator_var_snis / denominator_var_snis
        # An alternative formulation, which might be more intuitive if you think about
        # the "effective" number of samples and the variance of the weighted terms:
        # Var(mu_hat_snis) ~= (1/sum_weights^2) * Sum_i [ w_i^2 * (f(x_i) - mu_hat_snis)^2 ]
        # This is actually related to the formula used (especially for large N).
        # The (N/(N-1)) factor helps for smaller N.

        # Simpler version for large N (often used in practice):
        # var_mu_hat_snis_large_N = numerator_var_snis / denominator_var_snis
        
        se_mu_hat_snis = np.sqrt(var_mu_hat_snis)
        
        # Alternative common form for SE (often seen as an approximation):
        # se_mu_hat_snis_alt = np.sqrt(np.sum( ((weights / sum_weights) * (fx_values - mu_hat_snis))**2 ))
        # print(f"Alternative SE (SNIS): {se_mu_hat_snis_alt:.6f}") # Often very close
    else:
        se_mu_hat_snis = np.nan


# --- Effective Sample Size (ESS) ---
# ESS ≈ (Σ w_tilde_i)^2 / Σ (w_tilde_i^2)
# Here, w_tilde_i are our 'weights'
if sum_weights == 0 or np.sum(weights**2) == 0:
    ess = 0
else:
    ess = (sum_weights**2) / np.sum(weights**2)

# --- Output Results ---
true_value = p_mean**2 + p_std**2 # E_p[X^2] = mu_p^2 + sigma_p^2
print(f"--- Importance Sampling Results (N={N}) ---")
print(f"True Expected Value E_p[f(X)]: {true_value:.4f}")
print("-" * 40)

print("1. Unnormalized Importance Sampling:")
print(f"   Estimate (mu_hat_IS):      {mu_hat_is:.4f}")
print(f"   Standard Error (SE_IS):    {se_mu_hat_is:.4f}")
if se_mu_hat_is > 0:
    print(f"   Approx. 95% CI (IS):     [{mu_hat_is - 1.96*se_mu_hat_is:.4f}, {mu_hat_is + 1.96*se_mu_hat_is:.4f}]")
print("-" * 40)

print("2. Self-Normalized Importance Sampling:")
print(f"   Estimate (mu_hat_SNIS):    {mu_hat_snis:.4f}")
print(f"   Standard Error (SE_SNIS):  {se_mu_hat_snis:.4f}")
if not np.isnan(se_mu_hat_snis) and se_mu_hat_snis > 0:
    print(f"   Approx. 95% CI (SNIS):   [{mu_hat_snis - 1.96*se_mu_hat_snis:.4f}, {mu_hat_snis + 1.96*se_mu_hat_snis:.4f}]")
print("-" * 40)

print(f"Effective Sample Size (ESS): {ess:.2f} (out of {N} samples)")
if ess < N / 10:
    print("Warning: ESS is significantly lower than N. Weights might be highly skewed.")

#endregion -----------------------------------------------------------------------------------------
#region Tests 4 for Importance Sampling (Toy, 1D)

#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW # For weighted quantiles and CDF

# Set a seed for reproducibility
np.random.seed(42)

#%%
# --- 1. Define f(x) and p(x) ---
# def f(x):
#     """Target function to integrate."""
#     # return np.exp(-(x - 9)**2 / (2 * 0.5**2))
#     return stats.norm(8.8, 0.22).pdf(x)

# f(x) is peaked at c, zero elsewhere
p_lower_bound = 0
p_upper_bound = 10
c_val = 8.8 # Renamed from 'c' to avoid conflict with ECDF variable
f_peak_width = 0.5  # Controls how wide the non-zero part of f(x) is
f_active_half_width = 1.5 # f(x) will be non-zero in [c-f_active_half_width, c+f_active_half_width]

def f(x_arr):
    x_arr = np.asarray(x_arr)
    y = np.exp(-((x_arr - c_val)**2) / (2 * f_peak_width**2))
    condition = (x_arr >= p_lower_bound) & (x_arr <= p_upper_bound)
    return np.where(condition, y, 0.0)

#%% Define p(x)
# p(x) is U(0, 10)
p_lower_bound = 0
p_upper_bound = 10
def p_pdf(x):
    """PDF of the target distribution p(x) = U(0, 10)."""
    return stats.uniform.pdf(x, loc=p_lower_bound, scale=p_upper_bound - p_lower_bound)

def p_rvs(size):
    """Sample from p(x)."""
    return stats.uniform.rvs(loc=p_lower_bound, scale=p_upper_bound - p_lower_bound, size=size)

#%%
# --- 2. Ground Truth (using a very large number of MC samples) ---
N_ground_truth = 5 * 10**6
samples_gt = p_rvs(N_ground_truth)
f_values_gt = f(samples_gt)
ground_truth_mean = np.mean(f_values_gt)
ground_truth_std_err = np.std(f_values_gt) / np.sqrt(N_ground_truth)

print(f"--- Ground Truth Estimation (N={N_ground_truth}) ---")
print(f"Estimated Mean: {ground_truth_mean:.6f}")
print(f"Estimated Std Error of Mean: {ground_truth_std_err:.6f}")

# #%%
# # Analytical ground truth (for this specific f(x) and p(x))
# # I = (1/10) * integral_0^10 exp(-(x-9)^2/(2*0.5^2)) dx
# # Let y = (x-9)/0.5, dy = dx/0.5
# # I = (1/10) * 0.5 * integral_(-18)^(2) exp(-y^2/2) dy
# # I = (0.5/10) * sqrt(2*pi) * [Phi(2) - Phi(-18)]
# # where Phi is the CDF of standard normal.
# analytical_gt = (0.5/10) * np.sqrt(2*np.pi) * (stats.norm.cdf(2) - stats.norm.cdf(-18))
# print(f"Analytical Ground Truth Mean: {analytical_gt:.6f}\n")
# # Use analytical as the reference for true value.
# ground_truth_mean = analytical_gt

#%%
# --- 3. Monte Carlo (MC) Simulation ---
N_comparison = 1000  # Number of samples for MC and IS comparison

print(f"--- Standard Monte Carlo (N={N_comparison}) ---")
samples_mc = p_rvs(N_comparison)
f_values_mc = f(samples_mc)

mean_mc = np.mean(f_values_mc)
std_err_mc = np.std(f_values_mc, ddof=1) / np.sqrt(N_comparison) # ddof=1 for sample std dev

print(f"Mean (MC): {mean_mc:.6f}")
print(f"Std Error of Mean (MC): {std_err_mc:.6f}")

#%%
# --- 4. Importance Sampling (IS) with Truncated Normal ---
# Define proposal distribution q(x): Truncated Normal
q_mean = 8.8      # Center q where f(x) is large
q_std = 0.8       # Standard deviation of q
q_lower_bound = 0 # Truncation bounds for q (same as p's support)
q_upper_bound = 10

# Parameters for scipy.stats.truncnorm (a, b are for std normal)
a = (q_lower_bound - q_mean) / q_std
b = (q_upper_bound - q_mean) / q_std
q_dist = stats.truncnorm(a, b, loc=q_mean, scale=q_std)

def q_pdf(x):
    return q_dist.pdf(x)

def q_rvs(size):
    return q_dist.rvs(size)

print(f"\n--- Importance Sampling (N={N_comparison}) ---")
samples_is = q_rvs(N_comparison) # Samples from q(x)

# Ensure samples are within the support of p (they should be by q's truncation)
samples_is = np.clip(samples_is, p_lower_bound, p_upper_bound)

f_values_is = f(samples_is)
p_x_is = p_pdf(samples_is)
q_x_is = q_pdf(samples_is)

# Importance weights
weights_is = p_x_is / q_x_is
weights_is[q_x_is == 0] = 0 # Handle potential division by zero if q_pdf can be 0

# Self-normalized importance sampling estimator
mean_is = np.sum(weights_is * f_values_is) / np.sum(weights_is)

# Standard error for self-normalized IS
# Var_hat(μ_hat_SNIS) = Sum[(w_i * (f(x_i) - μ_hat_SNIS))^2] / (Sum[w_j])^2
var_is_num = np.sum((weights_is * (f_values_is - mean_is))**2)
var_is_den = (np.sum(weights_is))**2
var_is_estimator = var_is_num / var_is_den # This is the variance of the estimator
std_err_is = np.sqrt(var_is_estimator)

print(f"Mean (IS): {mean_is:.6f}")
print(f"Std Error of Mean (IS): {std_err_is:.6f}")

# Effective Sample Size (ESS)
ess_is = (np.sum(weights_is))**2 / np.sum(weights_is**2)
print(f"Effective Sample Size (ESS): {ess_is:.2f} (out of {N_comparison})")
print(f"Variance Reduction Factor (approx Var_MC/Var_IS): {(std_err_mc**2) / (std_err_is**2):.2f}")

#%%
# --- 5. Quantiles and their Standard Errors (using Bootstrap) ---
N_bootstrap = 1000
quantiles_to_calc = [0.25, 0.50, 0.75]

def bootstrap_quantiles(data, weights=None, quantiles=None, n_bootstrap=1000):
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    n_samples = len(data)
    bootstrap_quantile_estimates = np.zeros((n_bootstrap, len(quantiles)))

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled_data = data[indices]
        if weights is not None:
            resampled_weights = weights[indices]
            # Normalize weights for stability in DescrStatsW for each bootstrap sample
            # if np.sum(resampled_weights) > 1e-9: # Avoid division by zero if all weights are zero
            #     norm_resampled_weights = resampled_weights / np.sum(resampled_weights)
            # else:
            #     norm_resampled_weights = np.ones_like(resampled_weights) / len(resampled_weights)
            # No, DescrStatsW handles raw weights correctly.
            
            descr_stats = DescrStatsW(resampled_data, weights=resampled_weights, ddof=0)
            bootstrap_quantile_estimates[i, :] = descr_stats.quantile(quantiles)
        else:
            bootstrap_quantile_estimates[i, :] = np.quantile(resampled_data, quantiles)

    mean_q = np.mean(bootstrap_quantile_estimates, axis=0)
    std_err_q = np.std(bootstrap_quantile_estimates, axis=0, ddof=1)
    return mean_q, std_err_q # Returns means of bootstrapped quantiles and their SEs

# MC Quantiles
# For MC, we are interested in quantiles of f(X) where X ~ p(x)
# So we directly use f_values_mc
q_mc_direct = np.quantile(f_values_mc, quantiles_to_calc)
_, se_q_mc = bootstrap_quantiles(f_values_mc, quantiles=quantiles_to_calc, n_bootstrap=N_bootstrap)
print("\n--- Quantiles (MC) for f(X) where X ~ p(x) ---")
for q_val, q_direct, q_se in zip(quantiles_to_calc, q_mc_direct, se_q_mc):
    print(f"MC Q{int(q_val*100)}: {q_direct:.4f} (SE: {q_se:.4f})")


# IS Quantiles
# For IS, we want quantiles of f(X) where X ~ p(x), estimated using samples from q(x)
# We use f_values_is and weights_is
descr_stats_is = DescrStatsW(f_values_is, weights=weights_is, ddof=0) # ddof=0 for population quantiles
q_is_direct = descr_stats_is.quantile(quantiles_to_calc)
_, se_q_is = bootstrap_quantiles(f_values_is, weights=weights_is, quantiles=quantiles_to_calc, n_bootstrap=N_bootstrap)

print("\n--- Quantiles (IS) for f(X) where X ~ p(x) ---")
for q_val, q_direct, q_se in zip(quantiles_to_calc, q_is_direct, se_q_is):
    print(f"IS Q{int(q_val*100)}: {q_direct:.4f} (SE: {q_se:.4f})")

# Ground truth quantiles (from large MC run)
q_gt_direct = np.quantile(f_values_gt, quantiles_to_calc)
print("\n--- Quantiles (Ground Truth) for f(X) where X ~ p(x) ---")
for q_val, q_direct in zip(quantiles_to_calc, q_gt_direct):
    print(f"GT Q{int(q_val*100)}: {q_direct:.4f}")


# --- 6. Plot CDFs ---
# We want to plot the CDF of f(X) where X ~ p(x)

# Ground truth CDF (from large N MC samples)
stats_gt_cdf = DescrStatsW(f_values_gt, ddof=0) # No weights, effectively uniform
x_gt_cdf, y_gt_cdf = stats_gt_cdf.ecdf()

# MC CDF
stats_mc_cdf = DescrStatsW(f_values_mc, ddof=0) # No weights
x_mc_cdf, y_mc_cdf = stats_mc_cdf.ecdf()

# IS CDF (weighted)
stats_is_cdf = DescrStatsW(f_values_is, weights=weights_is, ddof=0)
x_is_cdf, y_is_cdf = stats_is_cdf.ecdf()


plt.figure(figsize=(12, 8))
plt.plot(x_gt_cdf, y_gt_cdf, label=f'Ground Truth CDF (N={N_ground_truth})', color='black', linestyle='--')
plt.plot(x_mc_cdf, y_mc_cdf, label=f'MC CDF (N={N_comparison})', color='blue', alpha=0.7)
plt.plot(x_is_cdf, y_is_cdf, label=f'IS CDF (N={N_comparison}, ESS={ess_is:.0f})', color='red', alpha=0.7)

plt.title('Cumulative Distribution Function of f(X)')
plt.xlabel('f(x) values')
plt.ylabel('CDF F(f(x))')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. Diagnostics Plots ---
plt.figure(figsize=(15, 5))

# Plot 1: f(x), p(x), q(x)
plt.subplot(1, 3, 1)
x_plot = np.linspace(0, 10, 500)
y_f = f(x_plot)
y_p = p_pdf(x_plot)
y_q = q_pdf(x_plot)

plt.plot(x_plot, y_f, label='f(x) (target function)', color='green')
plt.plot(x_plot, y_p, label='p(x) (target density U(0,10))', color='blue', linestyle=':')
plt.plot(x_plot, y_q, label='q(x) (proposal density)', color='red', linestyle='--')
# For scaling, normalize f(x) to fit on graph with densities
# plt.plot(x_plot, y_f / np.max(y_f) * np.max(y_q), label='f(x) (scaled)', color='green', linestyle='-.')

plt.title('Function and Densities')
plt.xlabel('x')
plt.ylabel('Value / Density')
plt.legend()
plt.grid(True)

# Plot 2: IS Weights
plt.subplot(1, 3, 2)
# Sort by sample value for a more intuitive plot, though scatter is fine too
sorted_indices_is = np.argsort(samples_is)
plt.scatter(samples_is[sorted_indices_is], weights_is[sorted_indices_is], alpha=0.5, s=10)
# For better visualization if weights vary a lot:
# plt.semilogy(samples_is[sorted_indices_is], weights_is[sorted_indices_is], 'o', alpha=0.5, markersize=3)
plt.title('Importance Weights w(x_i) = p(x_i)/q(x_i)')
plt.xlabel('x_i ~ q(x)')
plt.ylabel('Weight')
plt.grid(True)

# Plot 3: Product f(x)*p(x) and f(x)*q(x) (what we are sampling vs. what we want to sample)
plt.subplot(1, 3, 3)
plt.plot(x_plot, f(x_plot) * p_pdf(x_plot), label='f(x)p(x) (target integrand)', color='purple')
# Samples from q(x) weighted by f(x_i) gives an idea of where IS samples contribute most to the sum
# This is not f(x)q(x) directly, but related to the contribution f(x_i)p(x_i)/q(x_i) * q(x_i) = f(x_i)p(x_i)
# A better plot here is to show f(x)p(x) and where q(x) places samples.
# Overlap q(x) scaled to match the peak of f(x)p(x) might be informative
scale_factor_q = np.max(f(x_plot) * p_pdf(x_plot)) / np.max(q_pdf(x_plot))
plt.plot(x_plot, q_pdf(x_plot) * scale_factor_q, label='q(x) (scaled to match f(x)p(x) peak)', color='red', linestyle='--')

plt.title('Integrand f(x)p(x) and Scaled q(x)')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#endregion -----------------------------------------------------------------------------------------
#region Tests for Main Sampling

#%% Libraries
import dsplus as ds

from modules.compute_raster_stats import match_crs_to_raster
from modules.shift_storm_center import shift_gdf
from modules.compute_raster_stats import sum_raster_values_in_polygon
from modules.compute_depths import compute_depths
from modules.compute_prob_stats import print_sim_stats, get_df_freq_curve, get_prob

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
#region ARCHIVE Tests of Importance Sampling

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
#region ARCHIVE Random Tests

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
#region 

#%%
import platform
import pathlib

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
