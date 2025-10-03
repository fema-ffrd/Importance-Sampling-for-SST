import numpy as np
import pandas as pd

def metrics(
    obs: pd.DataFrame,
    sim: pd.DataFrame,
    rp_col: str = "RP",
    stats_map: dict | None = None,
    rp_min: int = 2,
    rp_max: int = 2000,
) -> pd.DataFrame:
    """
    Compare summarized depth curves (obs vs sim) over return periods and compute:
      - RMSE and Mean Error (sim - obs) for each statistic across all common RPs
      - Difference at the max shared RP (sim - obs) for each statistic

    Returns a single-row DataFrame.
    """
    if stats_map is None:
        stats_map = {
            "mean": "mean_in",
            "median": "median_in",
            "ci95_low": "ci95_low_in",
            "ci95_high": "ci95_high_in",
            "ci90_low": "ci90_low_in",
            "ci90_high": "ci90_high_in",
        }

    req = {rp_col, *stats_map.values()}
    miss_obs = req - set(obs.columns)
    miss_sim = req - set(sim.columns)
    if miss_obs:
        raise ValueError(f"`obs` missing: {sorted(miss_obs)}")
    if miss_sim:
        raise ValueError(f"`sim` missing: {sorted(miss_sim)}")

    # Restrict to RP window and inner-join on RP
    obs_use = obs[(obs[rp_col] >= rp_min) & (obs[rp_col] <= rp_max)][[rp_col, *stats_map.values()]].copy()
    sim_use = sim[(sim[rp_col] >= rp_min) & (sim[rp_col] <= rp_max)][[rp_col, *stats_map.values()]].copy()
    merged = pd.merge(obs_use, sim_use, on=rp_col, suffixes=("_obs", "_sim")).sort_values(rp_col)

    cols = ["n_points", "rp_min_used", "rp_max_used"]
    cols += [f"rmse_{k}" for k in stats_map]
    cols += [f"me_{k}" for k in stats_map]
    cols += [f"diff_at_maxrp_{k}" for k in stats_map]

    if merged.empty:
        return pd.DataFrame([[0, np.nan, np.nan] + [np.nan]*(len(cols)-3)], columns=cols)

    out = {
        "n_points": int(len(merged)),
        "rp_min_used": float(merged[rp_col].min()),
        "rp_max_used": float(merged[rp_col].max()),
    }

    # Metrics across all common RPs
    for label, col in stats_map.items():
        a = merged[f"{col}_sim"].astype(float).to_numpy()
        b = merged[f"{col}_obs"].astype(float).to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.any():
            err = a[m] - b[m]
            out[f"rmse_{label}"] = float(np.sqrt(np.mean(err**2)))
            out[f"me_{label}"]   = float(np.mean(err))
        else:
            out[f"rmse_{label}"] = np.nan
            out[f"me_{label}"]   = np.nan

    # Difference at max shared RP
    rp_max_shared = out["rp_max_used"]
    at_max = merged[merged[rp_col] == rp_max_shared]
    at_max = at_max.iloc[[-1]]
    for label, col in stats_map.items():
        sim_v = float(at_max[f"{col}_sim"].iloc[0]) if np.isfinite(at_max[f"{col}_sim"].iloc[0]) else np.nan
        obs_v = float(at_max[f"{col}_obs"].iloc[0]) if np.isfinite(at_max[f"{col}_obs"].iloc[0]) else np.nan
        out[f"diff_at_maxrp_{label}"] = sim_v - obs_v if (np.isfinite(sim_v) and np.isfinite(obs_v)) else np.nan

    return pd.DataFrame([out], columns=cols)
