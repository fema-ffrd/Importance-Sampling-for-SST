#region Libraries

#%%
import numpy as np
from scipy import stats
from scipy.optimize import brentq

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
        self.untruncated_dist = stats.gennorm(beta=self.beta, loc=self.loc, scale=self.scale)

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
        '''
        Probability density function for the truncated generalized normal.

        Args:
            v (float or np.ndarray): Value(s) at which to evaluate the PDF.

        Returns:
            np.ndarray: PDF values at v.
        '''
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
        '''
        Cumulative distribution function for the truncated generalized normal.

        Args:
            v (float or np.ndarray): Value(s) at which to evaluate the CDF.

        Returns:
            np.ndarray: CDF values at v.
        '''        
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
        '''
        Generate random samples from the truncated generalized normal distribution.

        Args:
            size (int): Number of samples to generate.

        Returns:
            np.ndarray: Random samples.
        '''      
        # Generate uniform samples in the range [CDF(m), CDF(n)]
        u_scaled = np.random.uniform(low=self.cdf_m, high=self.cdf_n, size=size)

        # Use the PPF (inverse CDF) of the untruncated distribution
        return self.untruncated_dist.ppf(u_scaled)

    def mean(self):
        '''
        Compute the mean of the truncated distribution.

        Returns:
            float: Mean value.
        '''     
        # Numerical integration for the mean of the truncated distribution
        # E[X] = integral from m to n of (x * pdf_truncated(x)) dx
        from scipy.integrate import quad
        integrand = lambda v: v * self.pdf(v)
        mean_val, _ = quad(integrand, self.m, self.n, limit=100) # Increased limit for potentially tricky integrals
        return mean_val

    def std(self):
        '''
        Compute the standard deviation of the truncated distribution.

        Returns:
            float: Standard deviation.
        '''       
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

#%% Truncated Distribution
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
        """Probability density function.
        Args:
            x (float or np.ndarray): Value(s) at which to evaluate the PDF.

        Returns:
            np.ndarray: PDF values at x.
        """
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
        """Cumulative distribution function.
        
        Args:
            x (float or np.ndarray): Value(s) at which to evaluate the CDF.

        Returns:
            np.ndarray: CDF values at x.
        
        """
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
        """Percent point function (inverse of cdf).
        Args:
            q (float or np.ndarray): Quantile(s) to evaluate.

        Returns:
            np.ndarray: PPF values at q.
        """
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
        """Random variates.

        Args:
            size (int): Number of random variates to generate.
            random_state (int or np.random.Generator, optional): Seed or random number generator for reproducibility.

        Returns:
            np.ndarray: Random variates sampled from the truncated distribution.
        """
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
        """
        Returns the support of the distribution (a, b).
        Returns:
            tuple: (a, b)        
        
        """
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
        """Calculates the mean of the truncated distribution.
        Returns:
            float: Mean value.                
        """
        if not self._moments_calculated:
            self._calculate_moments()
        return self._mean

    def var(self):
        """Calculates the variance of the truncated distribution.
        Returns:
            float: Variance value.
        """
        if not self._moments_calculated:
            self._calculate_moments()
        return self._var

    def __repr__(self):
        return f"TruncatedDistribution(a={self.a}, b={self.b})"

#%% Mixture Distribution
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
        """
        Probability density function.
         Args:
            x (float or np.ndarray): Value(s) at which to evaluate the PDF.

        Returns:
            np.ndarray: PDF values at x.          
        
        """
        x = np.asarray(x)
        pdf_val = (self.weight1 * self.dist1.pdf(x) +
                   self.weight2 * self.dist2.pdf(x))
        return pdf_val

    def cdf(self, x):
        """Cumulative distribution function.
        Args:
            x (float or np.ndarray): Value(s) at which to evaluate the CDF.

        Returns:
            np.ndarray: CDF values at x.
        """
        x = np.asarray(x)
        cdf_val = (self.weight1 * self.dist1.cdf(x) +
                   self.weight2 * self.dist2.cdf(x))
        return cdf_val

    def _solve_ppf_scalar(self, q_scalar):
        """
        Helper to solve PPF for a single scalar q using numerical root finding.
        
        Args:
            q_scalar (float): Quantile between 0 and 1.

        Returns:
            float: Value corresponding to the quantile.
        """
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
            return brentq(func_to_solve, a, b, xtol=1e-9, rtol=1e-9, maxiter=100)
        except ValueError:
            # This typically means f(a) and f(b) don't have opposite signs or other brentq issue
            # print(f"Warning: brentq failed for q={q_scalar:.4f}. Bounds [{a:.2e}, {b:.2e}], f(a)={fa:.2e}, f(b)={fb:.2e}.")
            # Fallback or raise error
            # A simple fallback could be nan or a weighted average of component ppfs (crude)
            if finite_ppf_vals:
                return self.weight1 * self.dist1.ppf(q_scalar) + self.weight2 * self.dist2.ppf(q_scalar) if len(finite_ppf_vals)==2 else finite_ppf_vals[0]
            return np.nan


    def ppf(self, q):
        """Percent point function (inverse of cdf).
        Args:
            q (float or np.ndarray): Quantile(s) to compute the PPF for.

        Returns:
            float or np.ndarray: The PPF value(s) corresponding to the input quantile(s).
        """
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
        """Random variates.
        Args:
            size (int): The number of random variates to generate.
            random_state (int or np.random.Generator, optional): Seed or random number generator for reproducibility.

        Returns:
            np.ndarray: Array of random variates sampled from the mixture distribution.
        """
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

#endregion -----------------------------------------------------------------------------------------
