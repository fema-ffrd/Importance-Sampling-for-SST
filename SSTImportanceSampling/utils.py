import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

def summarize_depths_by_return_period(
    df: pd.DataFrame,
    precip_col: str = "precip_avg_mm",
    exc_col: str = "exc_prb",
    realization_col: str = "realization",
    k: float = 10.0,
    rp_min: int = 2,
    rp_max_cap: int = 2000,
    use_common_min: bool = True,  # NEW: True -> min of maxima (common support), False -> max of maxima
) -> pd.DataFrame:
    """
    Build mean/median/CI bands (inches) of precipitation vs return period (years),
    and include the number of realizations and number of samples per realization
    (computed as total rows / n_realizations).

    Parameters
    ----------
    use_common_min : bool, default True
        If True, cap the RP grid at the *minimum* of the per-realization max RP
        (only the range that *all* realizations can cover).
        If False, cap at the *maximum* of the per-realization max RP
        (allow some realizations to be NaN beyond their max; aggregation uses nan-*).
    """
    need = {precip_col, exc_col, realization_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    work = df[[precip_col, exc_col, realization_col]].dropna(subset=[precip_col, exc_col]).copy()

    # RP per row via inverse of 1 - exp(-k * exc)
    work["RP"] = 1.0 / (1.0 - np.exp(-k * work[exc_col].astype(float)))

    # detect n_realizations and n_samples_per_realization directly
    n_realizations = df[realization_col].nunique()
    n_samples_per_realization = int(len(df) / n_realizations) if n_realizations > 0 else 0

    rp_max_each = []
    curves: dict = {}
    for r, g in work.groupby(realization_col, sort=False):
        g = g.copy()
        g = g[np.isfinite(g["RP"].values) & np.isfinite(g[precip_col].values)]
        if g.empty:
            continue

        g["precip_in"] = g[precip_col].astype(float) / 25.4
        # strictly increasing RP curve
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

    # Choose grid max: min of maxima (common support) or max of maxima (broader grid)
    per_realization_max = np.array(rp_max_each, dtype=float)
    if use_common_min:
        global_max = float(np.nanmin(per_realization_max))
    else:
        global_max = float(np.nanmax(per_realization_max))

    Rmax = int(min(rp_max_cap, np.floor(global_max)))
    if Rmax < rp_min:
        return pd.DataFrame(columns=[
            "RP","mean_in","median_in","ci90_low_in","ci90_high_in","ci95_low_in","ci95_high_in",
            "n_realizations","n_samples_per_realization"
        ])

    rp_grid = np.arange(rp_min, Rmax + 1, dtype=int)

    # Interpolate each realization onto the common grid; allow NaNs outside its range
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


def plot_return_period_summary(summary: pd.DataFrame, title: str = "Depth–Return Period Curve"):
    """
    Plot median and 95% confidence band of precipitation vs. return period.

    summary: DataFrame returned by summarize_depths_by_return_period
             must have columns ['RP','median_in','ci95_low_in','ci95_high_in']
    """
    if summary.empty:
        print("Summary DataFrame is empty. Nothing to plot.")
        return

    rp = summary["RP"].values
    med = summary["median_in"].values
    low = summary["ci95_low_in"].values
    high = summary["ci95_high_in"].values

    plt.figure(figsize=(9,6))
    plt.fill_between(rp, low, high, color="lightblue", alpha=0.4, label="95% CI")
    plt.plot(rp, med, color="blue", lw=2, label="Median")

    plt.xscale("log")

    # specify desired tick marks
    xticks = [2,5,10,25,50,100,200,500,1000,2000]
    plt.xticks(xticks, labels=[str(x) for x in xticks])

    plt.xlabel("Return Period (years)")
    plt.ylabel("Precipitation (inches)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_two_return_period_summaries(summary1: pd.DataFrame,
                                    summary2: pd.DataFrame,
                                    label1: str = "Uniform",
                                    label2: str = "Truncated Normal",
                                    title: str = "Depth–Return Period Curve (comparison)"):
    """
    Plot median + 95% CI for two summaries in one plot.

    Each summary must have columns:
      ['RP','median_in','ci95_low_in','ci95_high_in']
    """
    if summary1.empty or summary2.empty:
        print("One or both summary DataFrames are empty.")
        return

    rp1 = summary1["RP"].values
    med1 = summary1["median_in"].values
    low1 = summary1["ci95_low_in"].values
    high1 = summary1["ci95_high_in"].values

    rp2 = summary2["RP"].values
    med2 = summary2["median_in"].values
    low2 = summary2["ci95_low_in"].values
    high2 = summary2["ci95_high_in"].values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Fill + line for summary1
    ax.fill_between(rp1, low1, high1, color="lightblue", alpha=0.3)
    ax.plot(rp1, med1, color="blue", lw=2, label=f"{label1} (median)")

    # Fill + line for summary2
    ax.fill_between(rp2, low2, high2, color="lightcoral", alpha=0.3)
    ax.plot(rp2, med2, color="red", lw=2, label=f"{label2} (median)")

    # log x-axis with requested ticks
    ax.set_xscale("log")
    xticks = [2,5,10,25,50,100,200,500,1000,2000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Precipitation (inches)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # legend at bottom center
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9)

    ax.set_title(title, pad=12)
    fig.tight_layout()
    plt.show()


def plot_adaptive_evolution(history, watershed_gdf, domain_gdf, *, save=False, prefix="ais"):
    """
    Plot evolution of adaptive parameters.

    Parameters
    ----------
    history : pd.DataFrame
        Output from sampler.adapt(...). Must include columns:
        ['iter','mu_x_n','mu_y_n','sd_x_n','sd_y_n','mix','rho_n'].
        (Assumed to include an initial row with iter==0.)
    watershed_gdf, domain_gdf : GeoDataFrame
        Single-polygon GeoDataFrames in the same SHG CRS (meters).
    save : bool, default False
        If True, saves figures to disk with `prefix`.
    prefix : str, default "ais"
        Filename prefix when saving.
    """
    H = history.copy().sort_values("iter")
    it = H["iter"].to_numpy()

    # ---------- Figure 1: Map of narrow mean trajectory (meters) ----------
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2), dpi=300)
    domain_gdf.boundary.plot(ax=ax, linewidth=1.0)
    watershed_gdf.boundary.plot(ax=ax, linewidth=1.5)

    # Trajectory of narrow mean
    ax.plot(H["mu_x_n"], H["mu_y_n"], "-o", markersize=3, linewidth=1.0)

    # Annotate iteration numbers starting at 0
    for i, (x, y) in enumerate(zip(H["mu_x_n"], H["mu_y_n"])):
        ax.text(x, y, str(i), fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="0.6", lw=0.3, alpha=0.8))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    # no title per your request
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_map_narrow.png", bbox_inches="tight")

    # ---------- Figure 2: Parameters vs Iteration ----------
    fig2, axs = plt.subplots(2, 2, figsize=(7.2, 6.2), dpi=300)
    axs = axs.ravel()

    # Use discrete ticks at the observed iteration indices
    for ax in axs:
        ax.set_xticks(it)

    # Var_x (narrow) in m^2 (since sd is in meters)
    axs[0].plot(it, (H["sd_x_n"] ** 2), "-o", ms=3)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel(r"Variance $\sigma_x^2$ [m$^2$]")
    axs[0].grid(alpha=0.3, linewidth=0.6)

    # Var_y (narrow) in m^2
    axs[1].plot(it, (H["sd_y_n"] ** 2), "-o", ms=3)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel(r"Variance $\sigma_y^2$ [m$^2$]")
    axs[1].grid(alpha=0.3, linewidth=0.6)

    # Mixture weight β (unitless)
    axs[2].plot(it, H["mix"], "-o", ms=3)
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel(r"Mixture weight $\beta$ [–]")
    axs[2].grid(alpha=0.3, linewidth=0.6)

    # Copula correlation ρ (unitless)
    axs[3].plot(it, H["rho_n"], "-o", ms=3)
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylabel(r"Copula correlation $\rho$ [–]")
    axs[3].grid(alpha=0.3, linewidth=0.6)

    fig2.tight_layout()
    if save:
        fig2.savefig(f"{prefix}_param_evolution.png", bbox_inches="tight")

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