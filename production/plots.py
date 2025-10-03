import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

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