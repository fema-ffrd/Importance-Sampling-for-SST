from typing import Literal, Dict, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.stats import truncnorm, norm

class ImportanceSampler:
    """
    IS using multiple distributions over precomputed valid masks.

    Per realization:

    1. **Sample a storm** uniformly among storms with at least one valid cell.
    2. **Sample a valid cell** from that storm:
    3. **Compute deltas and importance weights**, then self-normalize.
    
    Supported distributions
    --------------------
    - ``"uniform"`` : cells sampled uniformly over the valid mask.
    - ``"truncated_gaussian"`` : cells sampled from a truncated Gaussian distribution restricted to the valid mask.
    - ``"gaussian_copula"`` : cells sampled using a Gaussian copula, allowing correlation between axes.
    - ``"mixture_trunc_gauss"`` : cells sampled from a weighted mixture of truncated Gaussians.

    """

    def __init__(
        self,
        distribution: Literal["uniform", "truncated_gaussian", "gaussian_copula", "mixture_trunc_gauss"],
        params: dict,
        num_simulations: int,
        num_realizations: int = 1,
        eps_floor: float = 1e-300,     # floor to keep q_base strictly positive
        use_alias: bool = False,       
    ):
        self.distribution = distribution
        self.params = dict(params or {})
        self.num_simulations = int(num_simulations)
        self.num_realizations = int(num_realizations)
        self.eps_floor = float(eps_floor)
        self.use_alias = bool(use_alias)
        self._validate_params()

    # ------------- validation -------------
    def _validate_params(self) -> None:
        d = self.distribution
        P = self.params

        def need(keys):
            missing = set(keys) - set(P.keys())
            if missing:
                raise ValueError(f"Missing required parameters for '{d}': {missing}")

        def pos(name):
            v = float(P[name]); 
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0.")
            return v

        if d == "uniform":
            return

        if d == "truncated_gaussian":
            need({"sd_x", "sd_y"})
            pos("sd_x"); pos("sd_y")
            return

        if d == "gaussian_copula":
            need({"sd_x", "sd_y", "rho"})
            pos("sd_x"); pos("sd_y")
            rho = float(P["rho"])
            if not (-0.999 < rho < 0.999):
                raise ValueError("rho must be in (-0.999, 0.999).")
            return

        if d == "mixture_trunc_gauss":
            need({"sd_x_narrow","sd_y_narrow","sd_x_wide","sd_y_wide","mix"})
            pos("sd_x_narrow"); pos("sd_y_narrow"); pos("sd_x_wide"); pos("sd_y_wide")
            mix = float(P["mix"])
            if not (0.0 < mix < 1.0):
                raise ValueError("mix must be in (0,1).")
            for key in ("rho","rho_narrow","rho_wide"):
                if key in P:
                    r = float(P[key])
                    if not (-0.999 < r < 0.999):
                        raise ValueError(f"{key} must be in (-0.999, 0.999).")
            return

        raise ValueError(f"Unsupported distribution: {d}")

    # ------------- build q_base grid -------------
    @staticmethod
    def _truncnorm_objs(bounds: Tuple[float,float,float,float], mx, my, sx, sy):
        xmin, ymin, xmax, ymax = bounds
        ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
        ay, by = (ymin - my)/sy, (ymax - my)/sy
        return truncnorm(ax, bx, loc=mx, scale=sx), truncnorm(ay, by, loc=my, scale=sy)

    def _qbase_trunc_gauss(self, xs: np.ndarray, ys: np.ndarray, bounds, mx, my, sx, sy) -> np.ndarray:
        tx, ty = self._truncnorm_objs(bounds, mx, my, sx, sy)
        fx = np.maximum(tx.pdf(xs), self.eps_floor)           # (nx,)
        fy = np.maximum(ty.pdf(ys), self.eps_floor)           # (ny,)
        return fy[:, None] * fx[None, :]                      # (ny, nx)

    def _qbase_gaussian_copula(self, xs, ys, bounds, mx, my, sx, sy, rho) -> np.ndarray:
        tx, ty = self._truncnorm_objs(bounds, mx, my, sx, sy)
        fx = np.maximum(tx.pdf(xs), self.eps_floor)
        fy = np.maximum(ty.pdf(ys), self.eps_floor)
        Fx = np.clip(tx.cdf(xs), 1e-15, 1-1e-15)
        Fy = np.clip(ty.cdf(ys), 1e-15, 1-1e-15)
        zx = norm.ppf(Fx)                                     # (nx,)
        zy = norm.ppf(Fy)                                     # (ny,)

        one_minus_r2 = 1.0 - rho*rho
        const = -0.5*np.log1p(-rho*rho)
        Zx = zx[None, :]                                      # (1,nx)
        Zy = zy[:, None]                                      # (ny,1)
        A = 0.5*(Zx**2 + Zy**2)
        B = 0.5*((Zx**2 - 2.0*rho*Zx*Zy + Zy**2) / one_minus_r2)
        qcop = np.exp(const + (A - B))
        q2d = qcop * (fy[:, None] * fx[None, :])
        return np.maximum(q2d, self.eps_floor)

    def _qbase_mixture(self, xs, ys, bounds, P) -> np.ndarray:
        mix = float(P["mix"])
        mxn = float(P.get("mu_x_narrow", P["mu_x"])); myn = float(P.get("mu_y_narrow", P["mu_y"]))
        mxw = float(P.get("mu_x_wide",   P["mu_x"])); myw = float(P.get("mu_y_wide",   P["mu_y"]))
        sxn = float(P["sd_x_narrow"]); syn = float(P["sd_y_narrow"])
        sxw = float(P["sd_x_wide"]);   syw = float(P["sd_y_wide"])
        rho_n = float(P.get("rho_narrow", P.get("rho", 0.0)))
        rho_w = float(P.get("rho_wide",  P.get("rho", 0.0)))
        q_n = self._qbase_gaussian_copula(xs, ys, bounds, mxn, myn, sxn, syn, rho_n)
        q_w = self._qbase_gaussian_copula(xs, ys, bounds, mxw, myw, sxw, syw, rho_w)
        return np.maximum(mix * q_n + (1.0 - mix) * q_w, self.eps_floor)

    def _build_q_base(self, mask_da: xr.DataArray, watershed_stats: Dict[str,float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = mask_da["x"].values
        ys = mask_da["y"].values
        bounds = (float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1]))

        # default mu to watershed centroid if not provided
        P = dict(self.params)
        if "mu_x" not in P or "mu_y" not in P:
            if watershed_stats is None or ("x" not in watershed_stats or "y" not in watershed_stats):
                raise ValueError("mu_x/mu_y not provided and watershed centroid unavailable.")
            P.setdefault("mu_x", float(watershed_stats["x"]))
            P.setdefault("mu_y", float(watershed_stats["y"]))

        if self.distribution == "uniform":
            q2d = np.ones((len(ys), len(xs)), dtype=float)
        elif self.distribution == "truncated_gaussian":
            q2d = self._qbase_trunc_gauss(xs, ys, bounds,
                                          mx=float(P["mu_x"]), my=float(P["mu_y"]),
                                          sx=float(P["sd_x"]), sy=float(P["sd_y"]))
        elif self.distribution == "gaussian_copula":
            q2d = self._qbase_gaussian_copula(xs, ys, bounds,
                                              mx=float(P["mu_x"]), my=float(P["mu_y"]),
                                              sx=float(P["sd_x"]), sy=float(P["sd_y"]),
                                              rho=float(P["rho"]))
        elif self.distribution == "mixture_trunc_gauss":
            q2d = self._qbase_mixture(xs, ys, bounds, P)
        else:
            raise RuntimeError("unreachable")

        return xs, ys, q2d

    # ------------- prepare valid sets -------------
    @staticmethod
    def _flatten_indices(mask2d: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask2d.astype(bool).ravel())

    def _prepare_catalog(self, data):
        """
        Returns:
          mask_da:   DataArray (storm,y,x)
          centers_df: filtered storm_centers
          per_storm: dict[name] -> {'flat_idx','count','qb_vals','Z'}
          xs, ys:    grid centers
          qb_flat:   q_base flattened (rows*cols,)
        """
        if not hasattr(data, "valid_mask_nc"):
            raise ValueError("`data` must have attribute `valid_mask_nc` (xarray DataArray).")
        if not hasattr(data, "storm_centers"):
            raise ValueError("`data` must have attribute `storm_centers` (DataFrame).")

        mask_da: xr.DataArray = data.valid_mask_nc
        if not {"storm","y","x"} <= set(mask_da.dims):
            raise ValueError("valid_mask_nc must have dims ('storm','y','x').")

        watershed_stats = getattr(data, "watershed_stats", None)
        xs, ys, q2d = self._build_q_base(mask_da, watershed_stats)
        qb_flat = q2d.ravel(order="C")

        storm_names_mask = mask_da["storm"].values.tolist()
        centers_df: pd.DataFrame = data.storm_centers.copy()
        need_cols = {"storm_path","x","y"}
        if not need_cols <= set(centers_df.columns):
            raise ValueError("storm_centers must contain columns ['storm_path','x','y'].")

        centers_df = centers_df[centers_df["storm_path"].astype(str).isin(storm_names_mask)].copy()
        if centers_df.empty:
            raise ValueError("No overlapping storms between valid_mask_nc and storm_centers.")
        centers_df["storm_path"] = centers_df["storm_path"].astype(str)

        per_storm = {}
        rows = mask_da.sizes["y"]; cols = mask_da.sizes["x"]
        for name in centers_df["storm_path"].unique():
            vm2d = mask_da.sel(storm=name).values.astype(bool)
            flat_idx = self._flatten_indices(vm2d)
            if flat_idx.size == 0:
                continue
            qb_vals = qb_flat[flat_idx]
            Z = float(qb_vals.sum()) if self.distribution != "uniform" else float(flat_idx.size)
            per_storm[name] = {
                "flat_idx": flat_idx,
                "count": flat_idx.size,
                "qb_vals": qb_vals,
                "Z": Z,
            }

        centers_df = centers_df[centers_df["storm_path"].isin(per_storm.keys())].reset_index(drop=True)
        if centers_df.empty:
            raise ValueError("All storms have empty valid sets; cannot sample.")

        return mask_da, centers_df, per_storm, xs, ys, qb_flat

    # ------------- public API -------------
    def sample(self, data) -> pd.DataFrame:
        """
        Inputs expected on `data`:
            - ``valid_mask_nc`` : xarray.DataArray (storm,y,x) with coords 'storm','x','y', values in {0,1}
            - ``storm_centers`` : DataFrame with ['storm_path','x','y']
            - ``watershed_stats`` : dict-like with ['x','y'] (centroid) used for default mu

        Returns
        -------
        :class:`pandas.DataFrame` with the following columns:

            - ``realization`` : realization index
            - ``realization_seed`` : random seed used for reproducibility
            - ``event_id`` : unique event identifier
            - ``storm_path`` : identifier or path of the storm
            - ``x, y`` : sampled storm-center coordinates
            - ``newx, newy`` : translated storm-center coordinates
            - ``delx, dely`` : translation deltas
            - ``weight_raw`` : unnormalized importance weight
            - ``weight`` : self-normalized importance weight
        """
        mask_da, centers_df, per_storm, xs, ys, qb_flat = self._prepare_catalog(data)

        storms = centers_df["storm_path"].values.astype(str)
        S = len(storms)
        s2xy = dict(zip(storms, zip(centers_df["x"].astype(float), centers_df["y"].astype(float))))

        all_frames = []

        # Create independent seeds for each realization and keep them
        seed_seq = np.random.SeedSequence()
        child_states = seed_seq.spawn(self.num_realizations)
        # Derive a single 64-bit seed value to store
        real_seeds = [int(ss.generate_state(1, dtype=np.uint64)[0]) for ss in child_states]

        for r_idx in range(self.num_realizations):
            rng = np.random.default_rng(real_seeds[r_idx])
            n = self.num_simulations

            # 1) storms uniformly
            chosen_idx = rng.integers(0, S, size=n)
            chosen_storms = storms[chosen_idx]

            # 2) cells per storm
            newx = np.empty(n, dtype=float)
            newy = np.empty(n, dtype=float)
            w_raw = np.empty(n, dtype=float)

            uniq, inv = np.unique(chosen_storms, return_inverse=True)
            rowsN = mask_da.sizes["y"]; colsN = mask_da.sizes["x"]

            for k, sname in enumerate(uniq):
                take = np.where(inv == k)[0]
                ps = per_storm[sname]
                flat_idx = ps["flat_idx"]

                if self.distribution == "uniform":
                    picks = rng.integers(0, flat_idx.size, size=take.size)
                    chosen_flat = flat_idx[picks]
                    w_raw[take] = 1.0
                else:
                    qb_vals = ps["qb_vals"]
                    Zj = ps["Z"]
                    p = qb_vals / (Zj if Zj > 0.0 else np.finfo(float).tiny)
                    chosen_idx_local = rng.choice(qb_vals.size, size=take.size, replace=True, p=p)
                    chosen_flat = flat_idx[chosen_idx_local]
                    q_at_p = qb_vals[chosen_idx_local]
                    w_raw[take] = Zj / (float(ps["count"]) * q_at_p)

                ii = chosen_flat // colsN
                jj = chosen_flat % colsN
                newx[take] = xs[jj]
                newy[take] = ys[ii]

            # 3) deltas
            ox = np.fromiter((s2xy[s][0] for s in chosen_storms), dtype=float, count=n)
            oy = np.fromiter((s2xy[s][1] for s in chosen_storms), dtype=float, count=n)
            delx = newx - ox
            dely = newy - oy

            # 4) normalize weights within realization
            if self.distribution == "uniform":
                w = np.full(n, 1.0 / n, dtype=float)
                w_raw_real = np.ones(n, dtype=float)
            else:
                w_raw_real = w_raw
                sum_w = float(np.sum(w_raw_real))
                if not np.isfinite(sum_w) or sum_w <= 0:
                    raise RuntimeError("Degenerate importance weights (sum <= 0). Check proposal parameters.")
                w = w_raw_real / sum_w

            out = pd.DataFrame({
                "realization": r_idx + 1,
                "realization_seed": real_seeds[r_idx],
                "event_id": np.arange(1, n+1, dtype=int),
                "storm_path": chosen_storms,
                "x": ox, "y": oy,
                "newx": newx, "newy": newy,
                "delx": delx, "dely": dely,
                "weight_raw": w_raw_real,
                "weight": w,
            })
            all_frames.append(out)

        return pd.concat(all_frames, ignore_index=True)
