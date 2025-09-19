import numpy as np
import pandas as pd

def summarize_depths_by_return_period(
    df: pd.DataFrame,
    precip_col: str = "precip_avg_mm",
    exc_col: str = "exc_prb",
    realization_col: str = "realization",
    k: float = 10.0,
    rp_min: int = 2,
    rp_max_cap: int = 2000,
) -> pd.DataFrame:
    """
    Build mean/median/CI bands (inches) of precipitation vs return period (years),
    and include the number of realizations and number of samples per realization
    (computed as total rows / n_realizations).
    """
    need = {precip_col, exc_col, realization_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    work = df[[precip_col, exc_col, realization_col]].dropna(subset=[precip_col, exc_col]).copy()

    # RP per row
    work["RP"] = 1.0 / (1.0 - np.exp(-k * work[exc_col].astype(float)))

    # detect n_realizations and n_samples_per_realization directly
    n_realizations = df[realization_col].nunique()
    n_samples_per_realization = int(len(df) / n_realizations) if n_realizations > 0 else 0

    rp_max_each = []
    curves = {}
    for r, g in work.groupby(realization_col, sort=False):
        g = g.copy()
        g = g[np.isfinite(g["RP"].values) & np.isfinite(g[precip_col].values)]
        if g.empty:
            continue

        g["precip_in"] = g[precip_col].astype(float) / 25.4
        g = g.sort_values("RP", ascending=True).drop_duplicates(subset="RP", keep="first")

        rp_vals = np.maximum.accumulate(g["RP"].to_numpy())
        keep = np.diff(np.r_[[-np.inf], rp_vals]) > 0
        rp_vals = rp_vals[keep]
        p_in    = g["precip_in"].to_numpy()[keep]

        if rp_vals.size == 0:
            continue

        rp_max_each.append(np.nanmax(rp_vals))
        curves[r] = (rp_vals, p_in)

    if not curves:
        return pd.DataFrame(columns=[
            "RP","mean_in","median_in","ci90_low_in","ci90_high_in","ci95_low_in","ci95_high_in",
            "n_realizations","n_samples_per_realization"
        ])

    global_max = float(np.nanmin(rp_max_each))
    Rmax = int(min(rp_max_cap, np.floor(global_max)))
    if Rmax < rp_min:
        return pd.DataFrame(columns=[
            "RP","mean_in","median_in","ci90_low_in","ci90_high_in","ci95_low_in","ci95_high_in",
            "n_realizations","n_samples_per_realization"
        ])

    rp_grid = np.arange(rp_min, Rmax + 1, dtype=int)

    interp_stack = [np.interp(rp_grid, rp, pin, left=np.nan, right=np.nan)
                    for rp, pin in curves.values()]
    interp_arr = np.vstack(interp_stack)

    mean_in   = np.nanmean(interp_arr, axis=0)
    median_in = np.nanmedian(interp_arr, axis=0)
    ci90_low_in, ci90_high_in = np.nanpercentile(interp_arr, [5.0, 95.0], axis=0)
    ci95_low_in, ci95_high_in = np.nanpercentile(interp_arr, [2.5, 97.5], axis=0)

    out = pd.DataFrame({
        "RP": rp_grid.astype(int),
        "mean_in": mean_in,
        "median_in": median_in,
        "ci90_low_in": ci90_low_in,
        "ci90_high_in": ci90_high_in,
        "ci95_low_in": ci95_low_in,
        "ci95_high_in": ci95_high_in,
        "n_realizations": n_realizations,
        "n_samples_per_realization": n_samples_per_realization,
    })

    stats_cols = ["mean_in","median_in","ci90_low_in","ci90_high_in","ci95_low_in","ci95_high_in"]
    return out.loc[~out[stats_cols].isna().all(axis=1)].reset_index(drop=True)
