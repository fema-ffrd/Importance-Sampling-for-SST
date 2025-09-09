from typing import Literal, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import uniform, truncnorm, norm, multivariate_normal

class ImportanceSampler:
    """
    Sampler for storm centers using various spatial distributions.

    Sampling logic per repetition:
      1) Uniformly (with replacement) pick storms from `data.storm_centers`
         (uses columns: storm_path, x, y).
      2) Draw (newx, newy) from the chosen spatial proposal over `data.domain_gdf`.
      3) Compute deltas: delx = newx - x, dely = newy - y.
      4) Attach SNIS weights from the proposal sampler.

    Output columns:
      ['rep','event_id','storm_path','x','y','newx','newy','delx','dely','weight']

    Supported distributions (proposal over absolute (newx,newy)):
        - 'uniform'              : Uniform over domain (weights uniform)
        - 'gaussian_copula'      : Truncated-normal marginals + Gaussian copula (ρ)
        - 'truncated_gaussian'   : Independent truncated normals
        - 'mixture_trunc_gauss'  : Two-component mixture of (possibly) correlated truncated normals

    Parameters (same meanings as before):
        distribution: Literal["uniform","gaussian_copula","truncated_gaussian","mixture_trunc_gauss"]
        params (dict) per distribution (see validation below)
        num_simulations: int  (per repetition)
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
                raise ValueError("rho must be in (-0.999, 0.999).")
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
                raise ValueError("mix must be in (0,1).")

            has_common = {"mu_x","mu_y"} <= P.keys()
            has_narrow = {"mu_x_narrow","mu_y_narrow"} <= P.keys()
            has_wide   = {"mu_x_wide","mu_y_wide"}     <= P.keys()
            if not ((has_narrow or has_common) and (has_wide or has_common)):
                raise ValueError(
                    "Provide (mu_x_narrow, mu_y_narrow) and (mu_x_wide, mu_y_wide), "
                    "or supply common (mu_x, mu_y)."
                )

            # Optional rhos
            for key in ("rho","rho_narrow","rho_wide"):
                if key in P:
                    r = float(P[key])
                    if not np.isfinite(r) or not (-0.999 < r < 0.999):
                        raise ValueError(f"{key} must be in (-0.999, 0.999).")
            return

        else:
            raise ValueError(f"Unsupported distribution: {d}")

    # ---------------- Public API ----------------
    def sample(self, data) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : object with attributes
            - data.domain_gdf : GeoDataFrame (single polygon)
            - data.storm_centers : DataFrame with columns ['storm_path','x','y']

        Returns
        -------
        DataFrame with:
        ['rep','event_id','storm_path','x','y','newx','newy','delx','dely','weight']
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        domain_gdf: gpd.GeoDataFrame = data.domain_gdf
        storm_centers: pd.DataFrame = data.storm_centers

        for c in ("storm_path","x","y"):
            if c not in storm_centers.columns:
                raise ValueError(f"`storm_centers` must contain column '{c}'")

        # Pick the sampler
        if self.distribution == "uniform":
            sampler = self._sample_uniform
        elif self.distribution == "truncated_gaussian":
            sampler = self._sample_truncated_gaussian
        elif self.distribution == "gaussian_copula":
            sampler = self._sample_gaussian_copula
        elif self.distribution == "mixture_trunc_gauss":
            sampler = self._sample_mixture_trunc_gauss
        else:
            raise RuntimeError("unreachable")

        # Draw spatial proposals once per rep
        spatial_df = sampler(domain_gdf)  # columns: rep,event_id,newx,newy,weight

        # For each rep, attach storms sampled uniformly WITH replacement
        results = []
        for rep in range(1, self.num_rep + 1):
            # rows for this rep from spatial_df
            S = spatial_df.loc[spatial_df["rep"] == rep].copy()
            n = len(S)
            # sample storms with replacement (uniform)
            chosen = storm_centers.sample(n=n, replace=True).reset_index(drop=True)
            S["storm_path"] = chosen["storm_path"].to_numpy()
            S["x"] = chosen["x"].astype(float).to_numpy()
            S["y"] = chosen["y"].astype(float).to_numpy()
            S["delx"] = S["newx"] - S["x"]
            S["dely"] = S["newy"] - S["y"]
            # reorder
            S = S[["rep","event_id","storm_path","x","y","newx","newy","delx","dely","weight"]]
            results.append(S)

        return pd.concat(results, ignore_index=True)

    # ---------------- Samplers (return newx,newy,weight) ----------------
    def _sample_uniform(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        poly = sp_domain.geometry.iloc[0]
        all_rows = []

        for rep in range(1, self.num_rep + 1):
            # oversample in bbox, then keep inside polygon
            x = uniform(xmin, xmax - xmin).rvs(self.num_simulations * 2)
            y = uniform(ymin, ymax - ymin).rvs(self.num_simulations * 2)
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=sp_domain.crs)
            inside = gdf.within(poly).to_numpy()
            x_in = x[inside][:self.num_simulations]
            y_in = y[inside][:self.num_simulations]

            if len(x_in) < self.num_simulations:
                # fallback loop to ensure count
                need = self.num_simulations - len(x_in)
                while need > 0:
                    x2 = uniform(xmin, xmax - xmin).rvs(need * 2)
                    y2 = uniform(ymin, ymax - ymin).rvs(need * 2)
                    g2 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x2, y2), crs=sp_domain.crs)
                    in2 = g2.within(poly).to_numpy()
                    xi = x2[in2][:need]
                    yi = y2[in2][:need]
                    x_in = np.concatenate([x_in, xi])
                    y_in = np.concatenate([y_in, yi])
                    need = self.num_simulations - len(x_in)

            # uniform target & proposal over polygon → equal weights
            w = np.full(self.num_simulations, 1.0 / self.num_simulations, dtype=float)

            all_rows.append(pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "newx": x_in,
                "newy": y_in,
                "weight": w,
            }))

        return pd.concat(all_rows, ignore_index=True)

    def _sample_truncated_gaussian(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)
        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        poly = sp_domain.geometry.iloc[0]

        P = self.params
        mu_x = float(P["mu_x"]); mu_y = float(P["mu_y"])
        sd_x = float(P["sd_x"]); sd_y = float(P["sd_y"])
        a_x, b_x = (xmin - mu_x)/sd_x, (xmax - mu_x)/sd_x
        a_y, b_y = (ymin - mu_y)/sd_y, (ymax - mu_y)/sd_y
        trunc_x = truncnorm(a_x, b_x, loc=mu_x, scale=sd_x)
        trunc_y = truncnorm(a_y, b_y, loc=mu_y, scale=sd_y)

        area = float(poly.area)
        eps = np.finfo(float).eps
        all_rows = []

        for rep in range(1, self.num_rep + 1):
            vx, vy, i = [], [], 0
            while len(vx) < self.num_simulations and i < 100:
                x = trunc_x.rvs(self.num_simulations)
                y = trunc_y.rvs(self.num_simulations)
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(poly)]
                vx += gdf.geometry.x.tolist()
                vy += gdf.geometry.y.tolist()
                i += 1
            vx = np.asarray(vx[:self.num_simulations]); vy = np.asarray(vy[:self.num_simulations])

            # SNIS weights: p(x,y)=1/area; q(x,y)=pdf_x * pdf_y (independent)
            qx = np.clip(trunc_x.pdf(vx), eps, np.inf)
            qy = np.clip(trunc_y.pdf(vy), eps, np.inf)
            w = (1.0 / area) / (qx * qy)
            w = w / w.sum()

            all_rows.append(pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "newx": vx,
                "newy": vy,
                "weight": w,
            }))

        return pd.concat(all_rows, ignore_index=True)

    def _sample_gaussian_copula(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)
        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        poly = sp_domain.geometry.iloc[0]

        P = self.params
        mu_x = float(P["mu_x"]); mu_y = float(P["mu_y"])
        sd_x = float(P["sd_x"]); sd_y = float(P["sd_y"])
        rho  = float(P["rho"])

        a_x, b_x = (xmin - mu_x)/sd_x, (xmax - mu_x)/sd_x
        a_y, b_y = (ymin - mu_y)/sd_y, (ymax - mu_y)/sd_y
        trunc_x = truncnorm(a_x, b_x, loc=mu_x, scale=sd_x)
        trunc_y = truncnorm(a_y, b_y, loc=mu_y, scale=sd_y)

        cov = np.array([[1.0, rho],[rho, 1.0]])
        L = np.linalg.cholesky(cov)

        area = float(poly.area)
        eps = np.finfo(float).eps
        all_rows = []

        for rep in range(1, self.num_rep + 1):
            vx, vy, i = [], [], 0
            while len(vx) < self.num_simulations and i < 100:
                # draw via Gaussian copula on latent normals
                u = np.clip(np.random.uniform(size=(self.num_simulations, 2)), eps, 1 - eps)
                z = norm.ppf(u)
                zc = z @ L.T
                uc = norm.cdf(zc)
                x = trunc_x.ppf(np.clip(uc[:, 0], eps, 1 - eps))
                y = trunc_y.ppf(np.clip(uc[:, 1], eps, 1 - eps))
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=sp_domain.crs)
                gdf = gdf[gdf.within(poly)]
                vx += gdf.geometry.x.tolist()
                vy += gdf.geometry.y.tolist()
                i += 1

            vx = np.asarray(vx[:self.num_simulations]); vy = np.asarray(vy[:self.num_simulations])

            # weights: q(x,y) = [ φ_Σ(z)/ (φ(zx)φ(zy)) ] * f_X(x) * f_Y(y)
            Fx = np.clip(trunc_x.cdf(vx), eps, 1 - eps)
            Fy = np.clip(trunc_y.cdf(vy), eps, 1 - eps)
            zx = norm.ppf(Fx); zy = norm.ppf(Fy)

            phi_bv = multivariate_normal(mean=[0, 0], cov=cov)
            q_cop = np.exp(phi_bv.logpdf(np.column_stack([zx, zy])) - (norm.logpdf(zx) + norm.logpdf(zy)))
            fx = np.clip(trunc_x.pdf(vx), eps, np.inf)
            fy = np.clip(trunc_y.pdf(vy), eps, np.inf)
            q = q_cop * fx * fy
            w = (1.0 / area) / q
            w = w / w.sum()

            all_rows.append(pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "newx": vx,
                "newy": vy,
                "weight": w,
            }))

        return pd.concat(all_rows, ignore_index=True)

    def _sample_mixture_trunc_gauss(self, sp_domain: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        xmin, ymin, xmax, ymax = sp_domain.total_bounds
        poly = sp_domain.geometry.iloc[0]
        area = float(poly.area)
        eps = np.finfo(float).eps

        P   = self.params
        # Means with fallback
        mxn = float(P.get("mu_x_narrow", P.get("mu_x"))); myn = float(P.get("mu_y_narrow", P.get("mu_y")))
        mxw = float(P.get("mu_x_wide",   P.get("mu_x"))); myw = float(P.get("mu_y_wide",   P.get("mu_y")))
        sxn, syn = float(P["sd_x_narrow"]), float(P["sd_y_narrow"])
        sxw, syw = float(P["sd_x_wide"]),   float(P["sd_y_wide"])
        mix = float(P["mix"])
        rho_n = float(P.get("rho", P.get("rho_narrow", 0.0)))
        rho_w = float(P.get("rho", P.get("rho_wide",   0.0)))

        # Truncations
        axn, bxn = (xmin - mxn)/sxn, (xmax - mxn)/sxn
        ayn, byn = (ymin - myn)/syn, (ymax - myn)/syn
        axw, bxw = (xmin - mxw)/sxw, (xmax - mxw)/sxw
        ayw, byw = (ymin - myw)/syw, (ymax - myw)/syw

        tx_n = truncnorm(axn, bxn, loc=mxn, scale=sxn)
        ty_n = truncnorm(ayn, byn, loc=myn, scale=syn)
        tx_w = truncnorm(axw, bxw, loc=mxw, scale=sxw)
        ty_w = truncnorm(ayw, byw, loc=myw, scale=syw)

        cov_n = np.array([[1.0, rho_n],[rho_n, 1.0]]); L_n = np.linalg.cholesky(cov_n)
        cov_w = np.array([[1.0, rho_w],[rho_w, 1.0]]); L_w = np.linalg.cholesky(cov_w)

        all_rows = []

        for rep in range(1, self.num_rep + 1):
            vx, vy, i = [], [], 0
            while len(vx) < self.num_simulations and i < 100:
                n = self.num_simulations
                choose_n = (np.random.rand(n) < mix)

                x = np.empty(n); y = np.empty(n)

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
                gdf = gdf[gdf.within(poly)]
                vx += gdf.geometry.x.tolist()
                vy += gdf.geometry.y.tolist()
                i += 1

            vx = np.asarray(vx[:self.num_simulations]); vy = np.asarray(vy[:self.num_simulations])

            # Compute mixture proposal density q(x,y) = mix*q_n + (1-mix)*q_w
            # Component latent maps
            Fx_n = np.clip(tx_n.cdf(vx), eps, 1 - eps); Fy_n = np.clip(ty_n.cdf(vy), eps, 1 - eps)
            zx_n = norm.ppf(Fx_n);       zy_n = norm.ppf(Fy_n)

            Fx_w = np.clip(tx_w.cdf(vx), eps, 1 - eps); Fy_w = np.clip(ty_w.cdf(vy), eps, 1 - eps)
            zx_w = norm.ppf(Fx_w);       zy_w = norm.ppf(Fy_w)

            phi_n = multivariate_normal(mean=[0,0], cov=cov_n)
            phi_w = multivariate_normal(mean=[0,0], cov=cov_w)

            # q_cop = φ_Σ(z) / [φ(zx) φ(zy)]
            qcop_n = np.exp(phi_n.logpdf(np.column_stack([zx_n, zy_n])) - (norm.logpdf(zx_n) + norm.logpdf(zy_n)))
            qcop_w = np.exp(phi_w.logpdf(np.column_stack([zx_w, zy_w])) - (norm.logpdf(zx_w) + norm.logpdf(zy_w)))

            fx_n = np.clip(tx_n.pdf(vx), eps, np.inf); fy_n = np.clip(ty_n.pdf(vy), eps, np.inf)
            fx_w = np.clip(tx_w.pdf(vx), eps, np.inf); fy_w = np.clip(ty_w.pdf(vy), eps, np.inf)

            q_n = qcop_n * fx_n * fy_n
            q_w = qcop_w * fx_w * fy_w
            q   = mix * q_n + (1.0 - mix) * q_w

            w = (1.0 / area) / q
            w = w / w.sum()

            all_rows.append(pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, self.num_simulations + 1),
                "newx": vx,
                "newy": vy,
                "weight": w,
            }))

        return pd.concat(all_rows, ignore_index=True)
