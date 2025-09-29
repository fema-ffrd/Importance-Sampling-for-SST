# consequence_adapt.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.stats import truncnorm, norm
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from affine import Affine

__all__ = [
    "AdaptParamsConsequence",
    "AdaptiveMixtureConsequenceSampler",
    "ConsequenceAdaptManager",
    "ensure_dir",
    "save_json",
    "load_json",
    "to_parquet",
    "read_parquet",
    "to_csv",
    "read_csv",
]

# --------- tiny helpers ----------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def to_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ----------------------------- params -----------------------------
@dataclass
class AdaptParamsConsequence:
    # Narrow (ADAPTED)
    mu_x_n: Optional[float] = None
    mu_y_n: Optional[float] = None
    sd_x_n: float = 1.0
    sd_y_n: float = 1.0
    rho_n: float = 0.0

    # Wide (FIXED)
    mu_x_w: Optional[float] = None
    mu_y_w: Optional[float] = None
    sd_x_w: float = 1.0
    sd_y_w: float = 1.0
    rho_w: float = 0.0

    # Mixture & smoothing
    mix: float = 0.30
    lambda_mix: float = 0.30
    beta_floor: float = 0.05
    beta_ceil: float = 0.95

    alpha: float = 0.75
    eps_var: float = 1e-6
    K_temper: float = 1.0

    beta_use_depth: bool = True  # kept for parity; not used directly in consequence mode

    # Numeric stability
    eps_floor: float = 1e-300


# -------------------------- helpers (q_base) --------------------------
def _truncnorm_objs(bounds: Tuple[float,float,float,float], mx, my, sx, sy):
    xmin, ymin, xmax, ymax = bounds
    ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
    ay, by = (ymin - my)/sy, (ymax - my)/sy
    return truncnorm(ax, bx, loc=mx, scale=sx), truncnorm(ay, by, loc=my, scale=sy)

def _qbase_gaussian_copula(xs, ys, bounds, mx, my, sx, sy, rho, eps_floor) -> np.ndarray:
    tx, ty = _truncnorm_objs(bounds, mx, my, sx, sy)
    fx = np.maximum(tx.pdf(xs), eps_floor)
    fy = np.maximum(ty.pdf(ys), eps_floor)
    Fx = np.clip(tx.cdf(xs), 1e-15, 1-1e-15)
    Fy = np.clip(ty.cdf(ys), 1e-15, 1-1e-15)
    zx = norm.ppf(Fx)
    zy = norm.ppf(Fy)

    one_minus_r2 = 1.0 - rho*rho
    const = -0.5*np.log1p(-rho*rho)
    Zx = zx[None, :]
    Zy = zy[:, None]
    A = 0.5*(Zx**2 + Zy**2)
    B = 0.5*((Zx**2 - 2.0*rho*Zx*Zy + Zy**2) / one_minus_r2)
    qcop = np.exp(const + (A - B))
    q2d = qcop * (fy[:, None] * fx[None, :])
    return np.maximum(q2d, eps_floor)


# ------------------------ main adaptive class ------------------------
class AdaptiveMixtureConsequenceSampler:
    """
    Samples valid cells per storm; computes watershed-avg precip for HMS CSV;
    and adapts once using externally computed 'consequence' as reward.

    Methods:
      - sample_batch(n, iter_idx, seed=None) -> DataFrame
      - depths_for(chosen_storms, chosen_flat) -> np.ndarray
      - adapt_once_from_dataframe(df, reward_col="consequence", threshold=0.0) -> dict
    """

    def __init__(self,
                 data,
                 params: AdaptParamsConsequence,
                 precip_cube: xr.DataArray | Dict[str, xr.DataArray] | None = None,
                 seed: Optional[int] = None,
                 align_precip: bool = True,
                 align_method: str = "nearest"):
        self.params = params
        self.rng = np.random.default_rng(seed)

        if not hasattr(data, "valid_mask_nc"):
            raise ValueError("`data` needs attribute `valid_mask_nc` (xr.DataArray).")
        if not hasattr(data, "storm_centers"):
            raise ValueError("`data` needs attribute `storm_centers` (DataFrame).")
        if not hasattr(data, "watershed_gdf"):
            raise ValueError("`data` needs attribute `watershed_gdf`.")

        self.mask_da: xr.DataArray = data.valid_mask_nc
        if not {"storm","y","x"} <= set(self.mask_da.dims):
            raise ValueError("valid_mask_nc dims must be ('storm','y','x').")

        self.xs = self.mask_da["x"].values
        self.ys = self.mask_da["y"].values
        self.colsN = len(self.xs)
        self.rowsN = len(self.ys)
        self.bounds = (float(self.xs[0]), float(self.ys[0]), float(self.xs[-1]), float(self.ys[-1]))

        self.dx = float(np.mean(np.diff(self.xs)))
        self.dy = float(np.mean(np.diff(self.ys)))
        self.transform = Affine.translation(self.xs[0] - self.dx/2.0, self.ys[0] - self.dy/2.0) * Affine.scale(self.dx, self.dy)

        ws_gdf: gpd.GeoDataFrame = data.watershed_gdf
        self.watershed_mask = geometry_mask(
            geometries=[mapping(geom) for geom in ws_gdf.geometry],
            out_shape=(self.rowsN, self.colsN),
            transform=self.transform,
            invert=True,
        )

        self.zero_pad_edges = True

        centers_df: pd.DataFrame = data.storm_centers.copy()
        need_cols = {"storm_path","x","y"}
        if not need_cols <= set(centers_df.columns):
            raise ValueError("storm_centers must have ['storm_path','x','y'].")
        storms_mask = self.mask_da["storm"].values.astype(str).tolist()
        centers_df["storm_path"] = centers_df["storm_path"].astype(str)
        centers_df = centers_df[centers_df["storm_path"].isin(storms_mask)].reset_index(drop=True)
        if centers_df.empty:
            raise ValueError("No overlapping storms between mask and centers.")
        self.centers_df = centers_df
        self.storms = centers_df["storm_path"].values.astype(str)
        self.s2i = {s: i for i, s in enumerate(self.storms)}
        self.orig_xy = centers_df.set_index("storm_path").loc[self.storms, ["x","y"]].to_numpy(float)

        ws_stats = getattr(data, "watershed_stats", {})
        if isinstance(ws_stats, pd.Series): ws_stats = ws_stats.to_dict()
        if self.params.mu_x_n is None: self.params.mu_x_n = float(ws_stats.get("x", np.nan))
        if self.params.mu_y_n is None: self.params.mu_y_n = float(ws_stats.get("y", np.nan))
        if self.params.mu_x_w is None: self.params.mu_x_w = float(ws_stats.get("x", np.nan))
        if self.params.mu_y_w is None: self.params.mu_y_w = float(ws_stats.get("y", np.nan))
        if not np.isfinite(self.params.mu_x_n) or not np.isfinite(self.params.mu_y_n):
            raise ValueError("Need narrow μ from watershed_stats[x,y] or set explicitly.")
        if not np.isfinite(self.params.mu_x_w) or not np.isfinite(self.params.mu_y_w):
            raise ValueError("Need wide μ from watershed_stats[x,y] or set explicitly.")

        self.per_storm: Dict[str, Dict[str, np.ndarray | float | int]] = {}
        for name in self.storms:
            m2d = self.mask_da.sel(storm=name).values.astype(bool)
            flat_idx = np.flatnonzero(m2d.ravel(order="C"))
            if flat_idx.size > 0:
                self.per_storm[name] = {"flat_idx": flat_idx, "count": flat_idx.size}
        if not self.per_storm:
            raise ValueError("All storms have empty valid regions.")

        self._rebuild_q_base()

        self.precip_cube = None
        raw_precip = precip_cube if precip_cube is not None else getattr(data, "cumulative_precip", None)
        if raw_precip is not None:
            self.precip_cube = self._normalize_and_align_precip(raw_precip, align=align_precip, method=align_method)

        self.history: List[dict] = []

    # ---------- q base ----------
    def _rebuild_q_base(self) -> None:
        p = self.params
        self.qn2d = _qbase_gaussian_copula(self.xs, self.ys, self.bounds, p.mu_x_n, p.mu_y_n, p.sd_x_n, p.sd_y_n, p.rho_n, p.eps_floor)
        self.qw2d = _qbase_gaussian_copula(self.xs, self.ys, self.bounds, p.mu_x_w, p.mu_y_w, p.sd_x_w, p.sd_y_w, p.rho_w, p.eps_floor)
        self.q2d  = np.maximum(p.mix * self.qn2d + (1.0 - p.mix) * self.qw2d, p.eps_floor)

        self.q_flat  = self.q2d.ravel(order="C")
        self.qn_flat = self.qn2d.ravel(order="C")
        self.qw_flat = self.qw2d.ravel(order="C")

        for name in self.per_storm.keys():
            ps = self.per_storm[name]
            qb_vals = self.q_flat[ps["flat_idx"]]
            Zj = float(qb_vals.sum())
            ps["qb_vals"] = qb_vals
            ps["Z"] = Zj

    # ---------- precip alignment ----------
    def _normalize_and_align_precip(self, raw, align: bool, method: str) -> xr.DataArray:
        def _ensure_da(da: xr.DataArray) -> xr.DataArray:
            dims = list(da.dims)
            if "storm" not in dims and "storm_path" in dims:
                da = da.rename({"storm_path": "storm"})
            if not {"storm","y","x"} <= set(da.dims):
                raise ValueError("precip DataArray needs dims ('storm','y','x').")
            if da["y"].values[0] > da["y"].values[-1]:
                da = da.sortby("y")
            if da["x"].values[0] > da["x"].values[-1]:
                da = da.sortby("x")
            return da.transpose("storm","y","x")

        if isinstance(raw, xr.Dataset):
            if "cumulative_precip" not in raw:
                raise ValueError("Dataset needs 'cumulative_precip'.")
            cube = _ensure_da(raw["cumulative_precip"])
        elif isinstance(raw, xr.DataArray):
            cube = _ensure_da(raw)
        elif isinstance(raw, dict):
            names, arrs, xs_ref, ys_ref = [], [], None, None
            for k, da in raw.items():
                if not isinstance(da, xr.DataArray) or ("x" not in da.dims or "y" not in da.dims):
                    raise ValueError("Each dict value must be xr.DataArray with ('y','x').")
                if da["y"].values[0] > da["y"].values[-1]: da = da.sortby("y")
                if da["x"].values[0] > da["x"].values[-1]: da = da.sortby("x")
                if xs_ref is None:
                    xs_ref, ys_ref = da["x"].values, da["y"].values
                else:
                    if not (np.array_equal(xs_ref, da["x"].values) and np.array_equal(ys_ref, da["y"].values)):
                        raise ValueError("All dict arrays must share identical x/y coords.")
                names.append(str(k)); arrs.append(da)
            cube = xr.concat(arrs, dim="storm").assign_coords(storm=np.array(names, dtype=str)).transpose("storm","y","x")
        else:
            raise TypeError("Unsupported precip type.")

        cube = cube.sel(storm=cube["storm"].astype(str).isin(self.storms))
        xs_p = cube["x"].values; ys_p = cube["y"].values
        same = np.array_equal(xs_p, self.xs) and np.array_equal(ys_p, self.ys)
        if not same and not align:
            raise ValueError("precip coords mismatch; set align_precip=True.")
        if not same:
            cube = cube.interp(x=("x", self.xs), y=("y", self.ys), method=method)
        return cube.transpose("storm","y","x")

    # ---------- sampling primitives ----------
    def _sample_valid_cells(self, n: int, rng: np.random.Generator):
        storms = self.storms
        S = storms.size
        chosen_idx = rng.integers(0, S, size=n)
        chosen_storms = storms[chosen_idx]

        chosen_flat = np.empty(n, dtype=int)
        w_raw = np.empty(n, dtype=float)

        uniq, inv = np.unique(chosen_storms, return_inverse=True)
        for k, sname in enumerate(uniq):
            take = np.where(inv == k)[0]
            ps = self.per_storm[sname]
            flat_idx = ps["flat_idx"]
            qb_vals = ps["qb_vals"]; Zj = ps["Z"]; cnt = ps["count"]

            p = qb_vals / (Zj if Zj > 0.0 else np.finfo(float).tiny)
            idx_local = rng.choice(qb_vals.size, size=take.size, replace=True, p=p)
            chosen_flat[take] = flat_idx[idx_local]
            q_at_p = qb_vals[idx_local]
            w_raw[take] = Zj / (float(cnt) * q_at_p)

        return chosen_storms, chosen_flat, w_raw

    def _flat_to_xy(self, flat_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ii = flat_idx // self.colsN
        jj = flat_idx % self.colsN
        return self.xs[jj], self.ys[ii]

    # ---------- precip depths for HMS CSV ----------
    def depths_for(self, chosen_storms: np.ndarray, chosen_flat: np.ndarray) -> np.ndarray:
        if self.precip_cube is None:
            return np.zeros(chosen_storms.size, dtype=float)

        depths = np.empty(chosen_storms.size, dtype=float)
        ii = chosen_flat // self.colsN
        jj = chosen_flat % self.colsN
        newx = self.xs[jj]
        newy = self.ys[ii]
        x0 = self.orig_xy[:, 0]
        y0 = self.orig_xy[:, 1]

        for n, s in enumerate(chosen_storms):
            k = self.s2i[str(s)]
            dx_cells = int(round((float(newx[n]) - float(x0[k])) / self.dx))
            dy_cells = int(round((float(newy[n]) - float(y0[k])) / self.dy))

            arr = self.precip_cube.isel(storm=k).values  # (rows, cols)
            shifted = np.roll(arr, shift=(dy_cells, dx_cells), axis=(0, 1))

            if self.zero_pad_edges:
                if dy_cells > 0:   shifted[:dy_cells, :] = 0.0
                elif dy_cells < 0: shifted[dy_cells:, :] = 0.0
                if dx_cells > 0:   shifted[:, :dx_cells] = 0.0
                elif dx_cells < 0: shifted[:, dx_cells:] = 0.0
            else:
                if dy_cells > 0:   shifted[:dy_cells, :] = np.nan
                elif dy_cells < 0: shifted[dy_cells:, :] = np.nan
                if dx_cells > 0:   shifted[:, :dx_cells] = np.nan
                elif dx_cells < 0: shifted[:, dx_cells:] = np.nan

            vals = shifted[self.watershed_mask]
            depths[n] = float(np.nan_to_num(vals, nan=0.0).mean()) if vals.size else np.nan

        return depths

    # ---------- PUBLIC: draw a batch for one iteration ----------
    def sample_batch(self, n: int, iter_idx: int, seed: Optional[int] = None) -> pd.DataFrame:
        rng = self.rng if seed is None else np.random.default_rng(seed)
        storms, flat_idx, w_raw = self._sample_valid_cells(n, rng)
        s = float(np.sum(w_raw))
        w = (w_raw / s) if (np.isfinite(s) and s > 0) else np.full_like(w_raw, 1.0 / len(w_raw))
        newx, newy = self._flat_to_xy(flat_idx)
        kidx = np.array([self.s2i[s] for s in storms], dtype=int)
        ox = self.orig_xy[kidx, 0]; oy = self.orig_xy[kidx, 1]
        depths = self.depths_for(storms, flat_idx)

        qn_at = self.qn_flat[flat_idx]
        qw_at = self.qw_flat[flat_idx]

        df = pd.DataFrame({
            "iter": iter_idx,
            "sample_id": [str(uuid.uuid4()) for _ in range(n)],
            "storm_path": storms,
            "x": ox, "y": oy,
            "newx": newx, "newy": newy,
            "delx": newx - ox, "dely": newy - oy,
            "flat_idx": flat_idx,
            "weight_raw": w_raw,
            "weight": w,
            "precip_avg_mm": depths,
            "qn_at": qn_at,
            "qw_at": qw_at,
        })
        return df

    # ---------- PUBLIC: one adaptation step from (samples + consequences) ----------
    def adapt_once_from_dataframe(self,
                                  df: pd.DataFrame,
                                  reward_col: str = "consequence",
                                  threshold: float = 0.0) -> dict:
        p = self.params
        if reward_col not in df.columns:
            raise ValueError(f"'{reward_col}' column missing in dataframe.")

        w_raw = df["weight_raw"].to_numpy(float)
        reward = df[reward_col].to_numpy(float)

        sumW = float(np.sum(w_raw))
        w_tilde = (w_raw / sumW) if (np.isfinite(sumW) and sumW > 0) else np.full_like(w_raw, 1.0/len(w_raw))

        ess = float(1.0 / np.sum(np.square(w_tilde)))
        with np.errstate(divide='ignore', invalid='ignore'):
            H = -np.nansum(w_tilde * np.log(np.maximum(w_tilde, 1e-300)))
        perplexity = float(np.exp(H) / len(w_tilde))

        qn_at = df["qn_at"].to_numpy(float)
        qw_at = df["qw_at"].to_numpy(float)
        denom = np.maximum(p.mix * qn_at + (1.0 - p.mix) * qw_at, p.eps_floor)
        r_n = (p.mix * qn_at) / denom

        rew = reward + p.K_temper
        rew = np.where(reward >= threshold, rew, 0.0)

        gamma = 1e-6
        num = float(np.sum(w_tilde * r_n * rew))
        den = float(np.sum(w_tilde * rew)) + gamma
        beta_em_depth = num / den if den > 0 else np.nan
        beta_em_uncond = float(np.sum(w_tilde * r_n))
        beta_used = beta_em_depth if np.isfinite(beta_em_depth) else beta_em_uncond
        beta_new = (1.0 - p.lambda_mix) * p.mix + p.lambda_mix * beta_used
        p.mix = float(np.clip(beta_new, p.beta_floor, p.beta_ceil))

        x_new = df["newx"].to_numpy(float)
        y_new = df["newy"].to_numpy(float)
        rb_w = w_tilde * r_n * rew
        s_rb = float(np.sum(rb_w))
        updated = 0
        if s_rb > 0 and np.count_nonzero(rew) > 10:
            rb_w /= s_rb
            mu_x_new = float(np.sum(rb_w * x_new))
            mu_y_new = float(np.sum(rb_w * y_new))
            var_x_new = float(np.sum(rb_w * (x_new - mu_x_new) ** 2) + p.eps_var)
            var_y_new = float(np.sum(rb_w * (y_new - mu_y_new) ** 2) + p.eps_var)

            p.mu_x_n = p.alpha * mu_x_new + (1 - p.alpha) * p.mu_x_n
            p.mu_y_n = p.alpha * mu_y_new + (1 - p.alpha) * p.mu_y_n
            vx = p.alpha * var_x_new + (1 - p.alpha) * (p.sd_x_n ** 2)
            vy = p.alpha * var_y_new + (1 - p.alpha) * (p.sd_y_n ** 2)
            p.sd_x_n = float(np.sqrt(max(vx, p.eps_var)))
            p.sd_y_n = float(np.sqrt(max(vy, p.eps_var)))

            eps = 1e-12
            tx_n, ty_n = _truncnorm_objs(self.bounds, p.mu_x_n, p.mu_y_n, p.sd_x_n, p.sd_y_n)
            u_x = np.clip(tx_n.cdf(x_new), eps, 1 - eps)
            u_y = np.clip(ty_n.cdf(y_new), eps, 1 - eps)
            z_x = norm.ppf(u_x); z_y = norm.ppf(u_y)

            mzx = float(np.sum(rb_w * z_x)); mzy = float(np.sum(rb_w * z_y))
            vzx = float(np.sum(rb_w * (z_x - mzx) ** 2) + p.eps_var)
            vzy = float(np.sum(rb_w * (z_y - mzy) ** 2) + p.eps_var)
            cov_zy = float(np.sum(rb_w * (z_x - mzx) * (z_y - mzy)))
            corr_hat = cov_zy / (np.sqrt(vzx) * np.sqrt(vzy))
            p.rho_n = float(np.clip(p.alpha * corr_hat + (1 - p.alpha) * p.rho_n, -0.95, 0.95))
            updated = 1

        self._rebuild_q_base()

        return dict(
            ess=float(ess),
            perplexity=float(perplexity),
            updated=int(updated),
            mix=float(p.mix),
            mu_x_n=float(p.mu_x_n), mu_y_n=float(p.mu_y_n),
            sd_x_n=float(p.sd_x_n), sd_y_n=float(p.sd_y_n),
            rho_n=float(p.rho_n),
            beta_em=float(beta_em_uncond),
            beta_em_depth=float(beta_em_depth) if np.isfinite(beta_em_depth) else np.nan,
            n=len(df),
        )


# --------------------- persistence/run manager ---------------------
class ConsequenceAdaptManager:
    """
    Disk layout:
      run_dir/
        run_meta.json
        params_iter_000.json
        history.csv
        iter_001/
          request_samples.parquet
          to_hms.csv
          consequences.csv
          merged.parquet
          params_after.json
        iter_002/
          ...
    """
    def __init__(self, sampler: AdaptiveMixtureConsequenceSampler, run_dir: str | Path, run_name: str, watershed_name: str):
        self.sampler = sampler
        self.run_dir = ensure_dir(Path(run_dir))
        self.run_name = run_name
        self.watershed_name = watershed_name

        meta_path = self.run_dir / "run_meta.json"
        if meta_path.exists():
            self.meta = load_json(meta_path)
        else:
            self.meta = dict(
                run_name=run_name,
                watershed=watershed_name,
                created=pd.Timestamp.utcnow().isoformat(),
            )
            save_json(self.meta, meta_path)

        p0 = self.run_dir / "params_iter_000.json"
        if not p0.exists():
            save_json(asdict(self.sampler.params), p0)

        self.hist_path = self.run_dir / "history.csv"
        if not self.hist_path.exists():
            pd.DataFrame(columns=[
                "iter","n","mix","mu_x_n","mu_y_n","sd_x_n","sd_y_n","rho_n",
                "ess","perplexity","beta_em","beta_em_depth","updated",
                "timestamp_utc"
            ]).to_csv(self.hist_path, index=False)

    def _iter_dir(self, iter_idx: int) -> Path:
        return ensure_dir(self.run_dir / f"iter_{iter_idx:03d}")

    def sample_and_write_request(self, iter_idx: int, samples_per_iter: int, seed: Optional[int]=None) -> dict:
        df = self.sampler.sample_batch(n=samples_per_iter, iter_idx=iter_idx, seed=seed)
        idir = self._iter_dir(iter_idx)
        req_pq = idir / "request_samples.parquet"
        df.to_parquet(req_pq, index=False)

        to_hms = df.loc[:, ["sample_id","storm_path","x","y","newx","newy","precip_avg_mm"]].copy()
        to_hms_csv = idir / "to_hms.csv"
        to_hms.to_csv(to_hms_csv, index=False)

        return dict(request_parquet=req_pq, to_hms_csv=to_hms_csv, count=len(df))

    def adapt_once_with_consequences(self,
                                     iter_idx: int,
                                     consequences_csv_path: str | Path,
                                     reward_col: str = "consequence",
                                     threshold: float = 0.0) -> dict:
        idir = self._iter_dir(iter_idx)
        req_pq = idir / "request_samples.parquet"
        if not req_pq.exists():
            raise FileNotFoundError(f"Missing request parquet for iter {iter_idx}: {req_pq}")

        req = pd.read_parquet(req_pq)
        cons = pd.read_csv(Path(consequences_csv_path))
        if "sample_id" not in cons.columns:
            raise ValueError("Consequence CSV must include 'sample_id' column to join.")
        merged = req.merge(cons, on="sample_id", how="inner", validate="one_to_one")
        if reward_col not in merged.columns:
            raise ValueError(f"Consequence CSV must include '{reward_col}' column.")

        snap = self.sampler.adapt_once_from_dataframe(merged, reward_col=reward_col, threshold=threshold)

        merged_pq = idir / "merged.parquet"
        merged.to_parquet(merged_pq, index=False)

        params_after = asdict(self.sampler.params)
        save_json(params_after, idir / "params_after.json")

        hist_row = dict(
            iter=iter_idx,
            n=snap["n"],
            mix=snap["mix"],
            mu_x_n=snap["mu_x_n"], mu_y_n=snap["mu_y_n"],
            sd_x_n=snap["sd_x_n"], sd_y_n=snap["sd_y_n"],
            rho_n=snap["rho_n"],
            ess=snap["ess"], perplexity=snap["perplexity"],
            beta_em=snap["beta_em"], beta_em_depth=snap["beta_em_depth"],
            updated=snap["updated"],
            timestamp_utc=pd.Timestamp.utcnow().isoformat(),
        )
        hist = pd.read_csv(self.hist_path)
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
        hist.to_csv(self.hist_path, index=False)

        return dict(merged_parquet=merged_pq, params_after=idir / "params_after.json", history_csv=self.hist_path)

    def current_params(self) -> AdaptParamsConsequence:
        return self.sampler.params
