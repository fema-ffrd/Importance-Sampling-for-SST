from typing import Literal, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from scipy.stats import uniform, truncnorm, norm, multivariate_normal

class ImportanceSampler:
    """
    One-sample-at-a-time validity:
      - For each selected storm (WITH replacement), keep drawing (newx,newy)
        from the chosen distribution until:
          (1) point is inside domain polygon, and
          (2) shifted watershed centroid (cx - dx, cy - dy) lies in valid_centroid_region_gdf.
      - Then compute deltas & weights.
    """

    def __init__(
        self,
        distribution: Literal["uniform", "gaussian_copula", "truncated_gaussian", "mixture_trunc_gauss"],
        params: dict,
        num_simulations: int,
        num_rep: int = 1,
        seed: Optional[int] = None,
        include_boundary: bool = False,
        max_iter_per_point: int = 10000,
    ):
        self.distribution = distribution
        self.params = params
        self.num_simulations = num_simulations
        self.num_rep = num_rep
        self.seed = seed
        self.include_boundary = include_boundary
        self.max_iter_per_point = max_iter_per_point
        self._validate_params()

    # ---------------- validation (unchanged logic) ----------------
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
            for key in ("rho","rho_narrow","rho_wide"):
                if key in P:
                    r = float(P[key])
                    if not np.isfinite(r) or not (-0.999 < r < 0.999):
                        raise ValueError(f"{key} must be in (-0.999, 0.999).")
            return

        else:
            raise ValueError(f"Unsupported distribution: {d}")

    # ---------------- helpers ----------------
    @staticmethod
    def _points_in(poly, xx, yy, include_boundary=False, crs=None):
        pts = gpd.GeoSeries(gpd.points_from_xy(xx, yy), crs=crs)
        return (pts.within(poly) | pts.touches(poly)) if include_boundary else pts.within(poly)

    def _propose_uniform(self, bounds, size):
        xmin, ymin, xmax, ymax = bounds
        x = uniform(xmin, xmax - xmin).rvs(size)
        y = uniform(ymin, ymax - ymin).rvs(size)
        return x, y

    def _propose_trunc_gauss_setup(self, bounds):
        P = self.params
        xmin, ymin, xmax, ymax = bounds
        mx, my = float(P["mu_x"]), float(P["mu_y"])
        sx, sy = float(P["sd_x"]), float(P["sd_y"])
        ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
        ay, by = (ymin - my)/sy, (ymax - my)/sy
        return truncnorm(ax, bx, loc=mx, scale=sx), truncnorm(ay, by, loc=my, scale=sy)

    def _propose_trunc_gauss(self, tx, ty, size):
        return tx.rvs(size), ty.rvs(size)

    def _propose_gauss_copula(self, tx, ty, rho, size):
        cov = np.array([[1.0, rho],[rho, 1.0]])
        L = np.linalg.cholesky(cov)
        eps = np.finfo(float).eps
        u = np.clip(np.random.uniform(size=(size,2)), eps, 1-eps)
        z = norm.ppf(u); zc = z @ L.T; uc = norm.cdf(zc)
        x = tx.ppf(np.clip(uc[:,0], eps, 1-eps))
        y = ty.ppf(np.clip(uc[:,1], eps, 1-eps))
        return x, y

    def _propose_mixture_setup(self, bounds):
        P = self.params
        xmin, ymin, xmax, ymax = bounds
        mxn = float(P.get("mu_x_narrow", P.get("mu_x"))); myn = float(P.get("mu_y_narrow", P.get("mu_y")))
        mxw = float(P.get("mu_x_wide",   P.get("mu_x"))); myw = float(P.get("mu_y_wide",   P.get("mu_y")))
        sxn, syn = float(P["sd_x_narrow"]), float(P["sd_y_narrow"])
        sxw, syw = float(P["sd_x_wide"]),   float(P["sd_y_wide"])
        axn, bxn = (xmin - mxn)/sxn, (xmax - mxn)/sxn
        ayn, byn = (ymin - myn)/syn, (ymax - myn)/syn
        axw, bxw = (xmin - mxw)/sxw, (xmax - mxw)/sxw
        ayw, byw = (ymin - myw)/syw, (ymax - myw)/syw
        tx_n = truncnorm(axn, bxn, loc=mxn, scale=sxn)
        ty_n = truncnorm(ayn, byn, loc=myn, scale=syn)
        tx_w = truncnorm(axw, bxw, loc=mxw, scale=sxw)
        ty_w = truncnorm(ayw, byw, loc=myw, scale=syw)
        rho_n = float(P.get("rho", P.get("rho_narrow", 0.0)))
        rho_w = float(P.get("rho", P.get("rho_wide",   0.0)))
        cov_n = np.array([[1.0, rho_n],[rho_n, 1.0]]); L_n = np.linalg.cholesky(cov_n)
        cov_w = np.array([[1.0, rho_w],[rho_w, 1.0]]); L_w = np.linalg.cholesky(cov_w)
        mix = float(P["mix"])
        return (tx_n, ty_n, L_n, tx_w, ty_w, L_w, mix)

    def _propose_mixture(self, setup, size):
        tx_n, ty_n, L_n, tx_w, ty_w, L_w, mix = setup
        eps = np.finfo(float).eps
        choose_n = (np.random.rand(size) < mix)
        x = np.empty(size); y = np.empty(size)
        if choose_n.any():
            u = np.clip(np.random.uniform(size=(choose_n.sum(),2)), eps, 1-eps)
            z = norm.ppf(u); zc = z @ L_n.T; uc = norm.cdf(zc)
            x[choose_n] = tx_n.ppf(np.clip(uc[:,0], eps, 1-eps))
            y[choose_n] = ty_n.ppf(np.clip(uc[:,1], eps, 1-eps))
        if (~choose_n).any():
            u = np.clip(np.random.uniform(size=((~choose_n).sum(),2)), eps, 1-eps)
            z = norm.ppf(u); zc = z @ L_w.T; uc = norm.cdf(zc)
            x[~choose_n] = tx_w.ppf(np.clip(uc[:,0], eps, 1-eps))
            y[~choose_n] = ty_w.ppf(np.clip(uc[:,1], eps, 1-eps))
        return x, y

    # ---------------- Public API ----------------
    def sample(self, data) -> pd.DataFrame:
        """
        Expects `data` to have:
          - domain_gdf
          - valid_centroid_region_gdf
          - storm_centers (cols: storm_path,x,y)
          - watershed_stats with keys 'x','y' (centroid)
        Returns columns:
          ['rep','event_id','storm_path','x','y','newx','newy','delx','dely','weight']
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        domain_gdf: gpd.GeoDataFrame = data.domain_gdf
        valid_centroid_region_gdf: gpd.GeoDataFrame = data.valid_centroid_region_gdf
        storm_centers: pd.DataFrame = data.storm_centers

        # watershed centroid
        if hasattr(data, "watershed_stats") and all(k in data.watershed_stats for k in ("x","y")):
            cx = float(data.watershed_stats["x"]); cy = float(data.watershed_stats["y"])
        else:
            # fallback to polygon centroid
            c = unary_union(domain_gdf.geometry).centroid
            cx, cy = float(c.x), float(c.y)

        for ccol in ("storm_path","x","y"):
            if ccol not in storm_centers.columns:
                raise ValueError(f"`storm_centers` must contain column '{ccol}'")

        # geometry setup
        D = unary_union(domain_gdf.geometry).buffer(0)
        V = unary_union(valid_centroid_region_gdf.geometry).buffer(0)
        if D.is_empty: raise ValueError("Domain geometry is empty.")
        if V.is_empty: raise ValueError("Valid centroid region is empty.")

        bounds = tuple(domain_gdf.total_bounds)
        area = float(D.area)
        eps = np.finfo(float).eps

        # choose storms WITH replacement for each rep
        all_out = []
        for rep in range(1, self.num_rep + 1):
            np.random.seed(self.seed)   #adding a new seed for each rep here. check later also need to store seeds
            chosen = storm_centers.sample(n=self.num_simulations, replace=True).reset_index(drop=True)
            ox = chosen["x"].to_numpy(dtype=float)
            oy = chosen["y"].to_numpy(dtype=float)
            sid = chosen["storm_path"].astype(str).to_numpy()

            # prepare proposers per distribution
            if self.distribution == "uniform":
                propose = lambda m: self._propose_uniform(bounds, m)
            elif self.distribution == "truncated_gaussian":
                tx, ty = self._propose_trunc_gauss_setup(bounds)
                propose = lambda m: self._propose_trunc_gauss(tx, ty, m)
            elif self.distribution == "gaussian_copula":
                tx, ty = self._propose_trunc_gauss_setup(bounds)
                rho = float(self.params["rho"])
                propose = lambda m: self._propose_gauss_copula(tx, ty, rho, m)
            elif self.distribution == "mixture_trunc_gauss":
                mix_setup = self._propose_mixture_setup(bounds)
                propose = lambda m: self._propose_mixture(mix_setup, m)
            else:
                raise RuntimeError("unreachable")

            # fill valid samples per-storm
            n = self.num_simulations
            newx = np.full(n, np.nan, dtype=float)
            newy = np.full(n, np.nan, dtype=float)
            pending = np.arange(n)
            tries = 0
            while pending.size and tries < self.max_iter_per_point:
                px, py = propose(pending.size)

                # inside domain polygon?
                inside = self._points_in(D, px, py, self.include_boundary, crs=domain_gdf.crs).to_numpy()

                # validity by shifted watershed centroid, for those candidates
                dx = px - ox[pending]
                dy = py - oy[pending]
                sx = cx - dx
                sy = cy - dy
                valid = self._points_in(V, sx, sy, self.include_boundary, crs=domain_gdf.crs).to_numpy()

                keep = inside & valid
                if keep.any():
                    idx_keep = pending[keep]
                    newx[idx_keep] = px[keep]
                    newy[idx_keep] = py[keep]
                    pending = pending[~keep]
                tries += 1

            if pending.size:
                raise RuntimeError(
                    f"Could not find valid centers for {pending.size} samples after {self.max_iter_per_point} tries. "
                    "Increase max_iter_per_point or adjust proposal/validity region."
                )

            # deltas
            delx = newx - ox
            dely = newy - oy

            # weights (same formulas as before)
            if self.distribution == "uniform":
                w = np.full(n, 1.0 / n, dtype=float)

            elif self.distribution == "truncated_gaussian":
                P = self.params
                xmin, ymin, xmax, ymax = bounds
                mx, my = float(P["mu_x"]), float(P["mu_y"])
                sx, sy = float(P["sd_x"]), float(P["sd_y"])
                ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
                ay, by = (ymin - my)/sy, (ymax - my)/sy
                tx = truncnorm(ax, bx, loc=mx, scale=sx)
                ty = truncnorm(ay, by, loc=my, scale=sy)
                qx = np.clip(tx.pdf(newx), eps, np.inf)
                qy = np.clip(ty.pdf(newy), eps, np.inf)
                q = qx * qy
                w = (1.0 / area) / q
                w = w / w.sum()

            elif self.distribution == "gaussian_copula":
                P = self.params
                xmin, ymin, xmax, ymax = bounds
                mx, my = float(P["mu_x"]), float(P["mu_y"])
                sx, sy = float(P["sd_x"]), float(P["sd_y"])
                rho = float(P["rho"])
                ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
                ay, by = (ymin - my)/sy, (ymax - my)/sy
                tx = truncnorm(ax, bx, loc=mx, scale=sx)
                ty = truncnorm(ay, by, loc=my, scale=sy)
                Fx = np.clip(tx.cdf(newx), eps, 1-eps)
                Fy = np.clip(ty.cdf(newy), eps, 1-eps)
                zx = norm.ppf(Fx); zy = norm.ppf(Fy)
                cov = np.array([[1.0, rho],[rho, 1.0]])
                phi = multivariate_normal(mean=[0,0], cov=cov)
                qcop = np.exp(phi.logpdf(np.column_stack([zx, zy])) - (norm.logpdf(zx) + norm.logpdf(zy)))
                fx = np.clip(tx.pdf(newx), eps, np.inf)
                fy = np.clip(ty.pdf(newy), eps, np.inf)
                q = qcop * fx * fy
                w = (1.0 / area) / q
                w = w / w.sum()

            elif self.distribution == "mixture_trunc_gauss":
                P = self.params
                xmin, ymin, xmax, ymax = bounds
                mxn = float(P.get("mu_x_narrow", P.get("mu_x"))); myn = float(P.get("mu_y_narrow", P.get("mu_y")))
                mxw = float(P.get("mu_x_wide",   P.get("mu_x"))); myw = float(P.get("mu_y_wide",   P.get("mu_y")))
                sxn, syn = float(P["sd_x_narrow"]), float(P["sd_y_narrow"])
                sxw, syw = float(P["sd_x_wide"]),   float(P["sd_y_wide"])
                mix = float(P["mix"])
                rho_n = float(P.get("rho", P.get("rho_narrow", 0.0)))
                rho_w = float(P.get("rho", P.get("rho_wide",   0.0)))

                axn, bxn = (xmin - mxn)/sxn, (xmax - mxn)/sxn
                ayn, byn = (ymin - myn)/syn, (ymax - myn)/syn
                axw, bxw = (xmin - mxw)/sxw, (xmax - mxw)/sxw
                ayw, byw = (ymin - myw)/syw, (ymax - myw)/syw

                tx_n = truncnorm(axn, bxn, loc=mxn, scale=sxn)
                ty_n = truncnorm(ayn, byn, loc=myn, scale=syn)
                tx_w = truncnorm(axw, bxw, loc=mxw, scale=sxw)
                ty_w = truncnorm(ayw, byw, loc=myw, scale=syw)

                cov_n = np.array([[1.0, rho_n],[rho_n, 1.0]])
                cov_w = np.array([[1.0, rho_w],[rho_w, 1.0]])
                phi_n = multivariate_normal(mean=[0,0], cov=cov_n)
                phi_w = multivariate_normal(mean=[0,0], cov=cov_w)

                Fx_n = np.clip(tx_n.cdf(newx), eps, 1-eps); Fy_n = np.clip(ty_n.cdf(newy), eps, 1-eps)
                zx_n = norm.ppf(Fx_n); zy_n = norm.ppf(Fy_n)
                Fx_w = np.clip(tx_w.cdf(newx), eps, 1-eps); Fy_w = np.clip(ty_w.cdf(newy), eps, 1-eps)
                zx_w = norm.ppf(Fx_w); zy_w = norm.ppf(Fy_w)

                qcop_n = np.exp(phi_n.logpdf(np.column_stack([zx_n, zy_n])) - (norm.logpdf(zx_n) + norm.logpdf(zy_n)))
                qcop_w = np.exp(phi_w.logpdf(np.column_stack([zx_w, zy_w])) - (norm.logpdf(zx_w) + norm.logpdf(zy_w)))

                fx_n = np.clip(tx_n.pdf(newx), eps, np.inf); fy_n = np.clip(ty_n.pdf(newy), eps, np.inf)
                fx_w = np.clip(tx_w.pdf(newx), eps, np.inf); fy_w = np.clip(ty_w.pdf(newy), eps, np.inf)

                q_n = qcop_n * fx_n * fy_n
                q_w = qcop_w * fx_w * fy_w
                q = mix * q_n + (1.0 - mix) * q_w
                w = (1.0 / area) / q
                w = w / w.sum()

            # assemble
            out = pd.DataFrame({
                "rep": rep,
                "event_id": np.arange(1, n + 1, dtype=int),
                "storm_path": sid,
                "x": ox,
                "y": oy,
                "newx": newx,
                "newy": newy,
                "delx": delx,
                "dely": dely,
                "weight": w,
            })
            all_out.append(out)

        return pd.concat(all_out, ignore_index=True)
