from typing import Literal, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import uniform, truncnorm, norm, multivariate_normal

class ImportanceSampler:
    """
    Sampler for storm centers using various spatial distributions.

    Supported distributions:
        - 'uniform'              : Uniform over domain (weights uniform)
        - 'gaussian_copula'      : Truncated-normal marginals + Gaussian copula (ρ)
        - 'truncated_gaussian'   : Independent truncated normals (no ρ)
        - 'mixture_trunc_gauss'  : Two-component mixture of independent truncated normals

    Parameters:
        distribution: Literal["uniform","gaussian_copula","truncated_gaussian","mixture_trunc_gauss"]
        params (dict):
          gaussian_copula:
            mu_x, mu_y, sd_x>0, sd_y>0, rho in (-1,1)
          truncated_gaussian:
            mu_x, mu_y, sd_x>0, sd_y>0
          mixture_trunc_gauss:
            mu_x, mu_y,
            sd_x_narrow>0, sd_y_narrow>0,
            sd_x_wide>0,  sd_y_wide>0,
            mix in (0,1)   # weight for "narrow" component
        num_simulations: int
        num_rep: int
        seed: Optional[int]
    """

    def __init__(
        self,
        distribution: Literal["uniform", "gaussian_copula", "truncated_gaussian", "mixture_trunc_gauss"],
        params: dict,
        num_simulations: int,
        num_rep: int = 1,
        seed: Optional[int] = None,
    ):
        self.distribution = distribution
        self.params = params
        self.num_simulations = num_simulations
        self.num_rep = num_rep
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        d = self.distribution
        P = self.params

        def need(keys):
            missing = set(keys) - P.keys()
            if missing:
                raise ValueError(f"Missing required parameters for '{d}': {missing}")

        def pos(name):
            v = float(P[name])
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite > 0 float.")
            return v

        if d == "uniform":
            return

        elif d == "gaussian_copula":
            need({"mu_x","mu_y","sd_x","sd_y","rho"})
            pos("sd_x"); pos("sd_y")
            rho = float(P["rho"])
            if not np.isfinite(rho) or not (-0.999 < rho < 0.999):
                raise ValueError("rho must be finite and strictly between -0.999 and 0.999.")
            return

        elif d == "truncated_gaussian":
            need({"mu_x","mu_y","sd_x","sd_y"})
            pos("sd_x"); pos("sd_y")
            return

        elif d == "mixture_trunc_gauss":
            need({"sd_x_narrow","sd_y_narrow","sd_x_wide","sd_y_wide","mix"})
            pos("sd_x_narrow"); pos("sd_y_narrow"); pos("sd_x_wide"); pos("sd_y_wide")
            mix = float(P["mix"])
            if not np.isfinite(mix) or not (0.0 < mix < 1.0):
                raise ValueError("mix must be finite and in (0,1).")

            # Means: per-component or common fallback
            has_common = {"mu_x","mu_y"} <= P.keys()
            has_narrow = {"mu_x_narrow","mu_y_narrow"} <= P.keys()
            has_wide   = {"mu_x_wide","mu_y_wide"}     <= P.keys()
            if not ((has_narrow or has_common) and (has_wide or has_common)):
                raise ValueError(
                    "Provide (mu_x_narrow, mu_y_narrow) and (mu_x_wide, mu_y_wide), "
                    "or supply common (mu_x, mu_y) as fallback."
                )

            # Optional correlations: rho (for both) or rho_narrow / rho_wide
            def chk_rho(name):
                if name in P:
                    r = float(P[name])
                    if not np.isfinite(r) or not (-0.999 < r < 0.999):
                        raise ValueError(f"{name} must be finite and in (-0.999, 0.999).")
            if "rho" in P:
                chk_rho("rho")
            else:
                # allow separate per-component rhos; if absent they default to 0 later
                if "rho_narrow" in P: chk_rho("rho_narrow")
                if "rho_wide"   in P: chk_rho("rho_wide")
            return

        else:
            raise ValueError(f"Unsupported distribution: {d}")


    def sample(self, domain_gdf: gpd.GeoDataFrame, watershed_gdf: Optional[gpd.GeoDataFrame]=None) -> pd.DataFrame:
        if self.distribution == "uniform":
            return self._sample_uniform(domain_gdf)
        if self.distribution == "gaussian_copula":
            return self._sample_gaussian_copula(domain_gdf)
        if self.distribution == "truncated_gaussian":
            return self._sample_truncated_gaussian(domain_gdf)
        if self.distribution == "mixture_trunc_gauss":
            return self._sample_mixture_trunc_gauss(domain_gdf)
        raise RuntimeError("unreachable")

    def _sample_uniform(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None: np.random.seed(self.seed)
        minx, miny, maxx, maxy = sp_domain.total_bounds
        dist_x = uniform(minx, maxx-minx)
        dist_y = uniform(miny, maxy-miny)
        all_samples = []
        for rep in range(1, self.num_rep+1):
            v_x, v_y, i = [], [], 0
            while len(v_x) < self.num_simulations and i < 100:
                _x = dist_x.rvs(self.num_simulations)
                _y = dist_y.rvs(self.num_simulations)
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_x,_y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(sp_domain.geometry.iloc[0])]
                v_x += gdf.geometry.x.tolist(); v_y += gdf.geometry.y.tolist(); i += 1
            v_x = v_x[:self.num_simulations]; v_y = v_y[:self.num_simulations]
            all_samples.append(pd.DataFrame({
                "rep": rep, "event_id": np.arange(1,self.num_simulations+1),
                "x": v_x, "y": v_y, "weight": np.full(self.num_simulations, 1.0/self.num_simulations)
            }))
        return pd.concat(all_samples, ignore_index=True)

    def _sample_truncated_gaussian(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None: np.random.seed(self.seed)
        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        mu_x = float(self.params["mu_x"]); mu_y = float(self.params["mu_y"])
        sd_x = float(self.params["sd_x"]); sd_y = float(self.params["sd_y"])
        a_x, b_x = (xmin-mu_x)/sd_x, (xmax-mu_x)/sd_x
        a_y, b_y = (ymin-mu_y)/sd_y, (ymax-mu_y)/sd_y
        trunc_x = truncnorm(a_x, b_x, loc=mu_x, scale=sd_x)
        trunc_y = truncnorm(a_y, b_y, loc=mu_y, scale=sd_y)

        area = sp_domain.geometry.iloc[0].area
        log_p = -np.log(area); eps = np.finfo(float).eps
        all_samples = []
        for rep in range(1, self.num_rep+1):
            v_x, v_y, i = [], [], 0
            while len(v_x) < self.num_simulations and i < 100:
                x = trunc_x.rvs(self.num_simulations); y = trunc_y.rvs(self.num_simulations)
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x,y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(sp_domain.geometry.iloc[0])]
                v_x += gdf.geometry.x.tolist(); v_y += gdf.geometry.y.tolist(); i += 1
            v_x = np.asarray(v_x[:self.num_simulations]); v_y = np.asarray(v_y[:self.num_simulations])

            log_q = np.log(np.clip(trunc_x.pdf(v_x), eps, np.inf)) + np.log(np.clip(trunc_y.pdf(v_y), eps, np.inf))
            w = np.exp(log_p - log_q); w /= w.sum()
            all_samples.append(pd.DataFrame({
                "rep": rep, "event_id": np.arange(1,self.num_simulations+1),
                "x": v_x, "y": v_y, "weight": w
            }))
        return pd.concat(all_samples, ignore_index=True)

    def _sample_gaussian_copula(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None: np.random.seed(self.seed)
        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        mu_x = float(self.params["mu_x"]); mu_y = float(self.params["mu_y"])
        sd_x = float(self.params["sd_x"]); sd_y = float(self.params["sd_y"])
        rho  = float(self.params["rho"])
        a_x, b_x = (xmin-mu_x)/sd_x, (xmax-mu_x)/sd_x
        a_y, b_y = (ymin-mu_y)/sd_y, (ymax-mu_y)/sd_y
        trunc_x = truncnorm(a_x, b_x, loc=mu_x, scale=sd_x)
        trunc_y = truncnorm(a_y, b_y, loc=mu_y, scale=sd_y)
        cov = np.array([[1.0, rho],[rho,1.0]]); L = np.linalg.cholesky(cov)
        area = sp_domain.geometry.iloc[0].area; log_p = -np.log(area)
        eps = np.finfo(float).eps; all_samples = []
        for rep in range(1, self.num_rep+1):
            v_x, v_y, i = [], [], 0
            while len(v_x) < self.num_simulations and i < 100:
                u = np.clip(np.random.uniform(size=(self.num_simulations,2)), eps, 1-eps)
                z = norm.ppf(u); zc = z @ L.T; uc = norm.cdf(zc)
                x = trunc_x.ppf(np.clip(uc[:,0], eps, 1-eps))
                y = trunc_y.ppf(np.clip(uc[:,1], eps, 1-eps))
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x,y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(sp_domain.geometry.iloc[0])]
                v_x += gdf.geometry.x.tolist(); v_y += gdf.geometry.y.tolist(); i += 1
            v_x = np.asarray(v_x[:self.num_simulations]); v_y = np.asarray(v_y[:self.num_simulations])

            cdf_x = np.clip(trunc_x.cdf(v_x), eps, 1-eps); cdf_y = np.clip(trunc_y.cdf(v_y), eps, 1-eps)
            zx = norm.ppf(cdf_x); zy = norm.ppf(cdf_y)
            phi_bv = multivariate_normal(mean=[0,0], cov=cov)
            log_q_lat = phi_bv.logpdf(np.column_stack([zx,zy]))
            log_phi = norm.logpdf(zx) + norm.logpdf(zy)
            log_fx = np.log(np.clip(trunc_x.pdf(v_x), eps, np.inf))
            log_fy = np.log(np.clip(trunc_y.pdf(v_y), eps, np.inf))
            log_q = (log_q_lat - log_phi) + (log_fx + log_fy)
            w = np.exp(log_p - log_q); s = w.sum(); w = np.full_like(w, 1/len(w)) if (s==0 or not np.isfinite(s)) else (w/s)
            all_samples.append(pd.DataFrame({
                "rep": rep, "event_id": np.arange(1,self.num_simulations+1),
                "x": v_x, "y": v_y, "weight": w
            }))
        return pd.concat(all_samples, ignore_index=True)

    def _sample_mixture_trunc_gauss(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Two-component mixture of rectangle-truncated normals with DISTINCT means
        and per-component correlation via a Gaussian copula.

        Components (each c ∈ {narrow, wide}):
            X_c ~ TN(mu_x_c, sd_x_c; [xmin,xmax])
            Y_c ~ TN(mu_y_c, sd_y_c; [ymin,ymax])
            Dependence: Corr(Zx,Zy)=rho_c on latent standard normals, then
            (X_c, Y_c) obtained by inverse CDF of truncated marginals.

        Proposal density for a component c:
            q_c(x,y) = [ φ_Σc(z) / (φ(zx) φ(zy)) ] * f_Xc(x) * f_Yc(y)
        where z = (Φ^{-1}(F_Xc(x)), Φ^{-1}(F_Yc(y))), Σc = [[1,ρc],[ρc,1]]

        Mixture proposal:
            q(x,y) = mix * q_n(x,y) + (1-mix) * q_w(x,y)

        Target (uniform on polygon): p(x,y) = 1/area
        Weights (normalized per repetition): w ∝ p / q
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        area = float(sp_domain.geometry.iloc[0].area)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Domain polygon area must be positive and finite.")
        log_p = -np.log(area)
        eps = np.finfo(float).eps

        P   = self.params
        # Means with fallback
        mxn = float(P.get("mu_x_narrow", P.get("mu_x"))); myn = float(P.get("mu_y_narrow", P.get("mu_y")))
        mxw = float(P.get("mu_x_wide",   P.get("mu_x"))); myw = float(P.get("mu_y_wide",   P.get("mu_y")))
        if any(v is None for v in (mxn, myn, mxw, myw)):
            raise ValueError("Provide (mu_x_narrow, mu_y_narrow) and (mu_x_wide, mu_y_wide), "
                            "or common (mu_x, mu_y) fallback.")

        # Scales and mixture weight
        sxn, syn = float(P["sd_x_narrow"]), float(P["sd_y_narrow"])
        sxw, syw = float(P["sd_x_wide"]),   float(P["sd_y_wide"])
        mix = float(P["mix"])

        # Correlations (optional). If 'rho' given, use for both; else per-component with default 0.
        rho_n = float(P.get("rho", P.get("rho_narrow", 0.0)))
        rho_w = float(P.get("rho", P.get("rho_wide",   0.0)))

        # Build truncated marginals
        a_xn, b_xn = (xmin - mxn)/sxn, (xmax - mxn)/sxn
        a_yn, b_yn = (ymin - myn)/syn, (ymax - myn)/syn
        a_xw, b_xw = (xmin - mxw)/sxw, (xmax - mxw)/sxw
        a_yw, b_yw = (ymin - myw)/syw, (ymax - myw)/syw

        tx_n = truncnorm(a_xn, b_xn, loc=mxn, scale=sxn)
        ty_n = truncnorm(a_yn, b_yn, loc=myn, scale=syn)
        tx_w = truncnorm(a_xw, b_xw, loc=mxw, scale=sxw)
        ty_w = truncnorm(a_yw, b_yw, loc=myw, scale=syw)

        # Cholesky factors for component copulas
        cov_n = np.array([[1.0, rho_n],[rho_n, 1.0]]); L_n = np.linalg.cholesky(cov_n)
        cov_w = np.array([[1.0, rho_w],[rho_w, 1.0]]); L_w = np.linalg.cholesky(cov_w)

        all_samples = []

        for rep in range(1, self.num_rep + 1):
            v_x, v_y = [], []
            i, max_iter = 0, 100

            # Draw until N points are inside polygon
            while len(v_x) < self.num_simulations and i < max_iter:
                n = self.num_simulations

                # Allocate components
                choose_n = (np.random.rand(n) < mix)
                x = np.empty(n); y = np.empty(n)

                # Sample latent copula normals and map via marginals for each component
                if choose_n.any():
                    u = np.clip(np.random.uniform(size=(choose_n.sum(), 2)), eps, 1 - eps)
                    z = norm.ppf(u); zc = z @ L_n.T; uc = norm.cdf(zc)
                    x[choose_n] = tx_n.ppf(np.clip(uc[:,0], eps, 1-eps))
                    y[choose_n] = ty_n.ppf(np.clip(uc[:,1], eps, 1-eps))

                if (~choose_n).any():
                    u = np.clip(np.random.uniform(size=((~choose_n).sum(), 2)), eps, 1 - eps)
                    z = norm.ppf(u); zc = z @ L_w.T; uc = norm.cdf(zc)
                    x[~choose_n] = tx_w.ppf(np.clip(uc[:,0], eps, 1-eps))
                    y[~choose_n] = ty_w.ppf(np.clip(uc[:,1], eps, 1-eps))

                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(sp_domain.geometry.iloc[0])]
                v_x += gdf.geometry.x.tolist()
                v_y += gdf.geometry.y.tolist()
                i += 1

            v_x = np.asarray(v_x[:self.num_simulations])
            v_y = np.asarray(v_y[:self.num_simulations])

            # Compute component log-densities q_n and q_w at (v_x, v_y)
            # Step 1: map to latent normals per component
            Fx_n = np.clip(tx_n.cdf(v_x), eps, 1 - eps); Fy_n = np.clip(ty_n.cdf(v_y), eps, 1 - eps)
            zx_n = norm.ppf(Fx_n); zy_n = norm.ppf(Fy_n)
            Fx_w = np.clip(tx_w.cdf(v_x), eps, 1 - eps); Fy_w = np.clip(ty_w.cdf(v_y), eps, 1 - eps)
            zx_w = norm.ppf(Fx_w); zy_w = norm.ppf(Fy_w)

            # Step 2: copula terms φ_Σ(z) and φ(z)
            phi_n = multivariate_normal(mean=[0,0], cov=cov_n)
            phi_w = multivariate_normal(mean=[0,0], cov=cov_w)
            log_phi_n = phi_n.logpdf(np.column_stack([zx_n, zy_n]))
            log_phi_w = phi_w.logpdf(np.column_stack([zx_w, zy_w]))
            log_phi_std_n = norm.logpdf(zx_n) + norm.logpdf(zy_n)
            log_phi_std_w = norm.logpdf(zx_w) + norm.logpdf(zy_w)

            # Step 3: marginal truncated densities
            log_fx_n = np.log(np.clip(tx_n.pdf(v_x), eps, np.inf)); log_fy_n = np.log(np.clip(ty_n.pdf(v_y), eps, np.inf))
            log_fx_w = np.log(np.clip(tx_w.pdf(v_x), eps, np.inf)); log_fy_w = np.log(np.clip(ty_w.pdf(v_y), eps, np.inf))

            # Component log q's
            log_q_n = (log_phi_n - log_phi_std_n) + (log_fx_n + log_fy_n)
            log_q_w = (log_phi_w - log_phi_std_w) + (log_fx_w + log_fy_w)

            # Mixture log q via log-sum-exp
            a = np.log(mix)     + log_q_n
            b = np.log(1.0-mix) + log_q_w
            m = np.maximum(a, b)
            log_q = m + np.log(np.exp(a - m) + np.exp(b - m))

            # Weights
            w = np.exp(log_p - log_q)
            w_sum = w.sum()
            w = (np.full_like(w, 1.0/len(w)) if (w_sum == 0 or not np.isfinite(w_sum)) else (w / w_sum))

            all_samples.append(pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "x": v_x,
                "y": v_y,
                "weight": w,
            }))

        return pd.concat(all_samples, ignore_index=True)
