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
                                     title: str = "Depth–Return Period Curve (comparison)",
                                     save=False):
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

    if save:
        fig.savefig(f"{title}.png", bbox_inches="tight")

def plot_adaptive_evolution(history, watershed_gdf, domain_gdf, *, save=False, prefix="ais"):
    """
    history: DataFrame from sampler.adapt(...)
    watershed_gdf/domain_gdf: single-polygon GeoDataFrames in same CRS as sampler
    """
    H = history.copy().sort_values("iter")
    it = H["iter"].to_numpy()

    # ---------- Figure 1: MAP of narrow mean trajectory ----------
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    domain_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    watershed_gdf.boundary.plot(ax=ax, linewidth=1.5, color="blue")

    # Plot trajectory of narrow mean
    ax.plot(H["mu_x_n"], H["mu_y_n"], "-o", markersize=3, linewidth=1.0, label="Narrow μ")

    # Annotate iteration numbers
    for i, (x, y) in enumerate(zip(H["mu_x_n"], H["mu_y_n"]), start=1):
        ax.text(x, y, str(i), fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="gray", lw=0.3, alpha=0.7))

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Adaptive narrow mean trajectory", pad=6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_map_narrow.png", bbox_inches="tight")

    # ---------- Figure 2: Parameters vs iteration ----------
    fig2, axs = plt.subplots(2, 2, figsize=(6, 5.5), dpi=300)
    axs = axs.ravel()

    axs[0].plot(it, H["sd_x_n"]**2, "-o", ms=3)
    axs[0].set_title("Var$_x$ (narrow)"); axs[0].set_xlabel("Iter"); axs[0].set_ylabel("sd$_x^2$")

    axs[1].plot(it, H["sd_y_n"]**2, "-o", ms=3)
    axs[1].set_title("Var$_y$ (narrow)"); axs[1].set_xlabel("Iter"); axs[1].set_ylabel("sd$_y^2$")

    axs[2].plot(it, H["mix"], "-o", ms=3)
    axs[2].set_title("Mixture weight"); axs[2].set_xlabel("Iter"); axs[2].set_ylabel("mix")

    if "hit_rate_weighted" in H:
        axs[3].plot(it, H["hit_rate_weighted"], "-o", ms=3)
        axs[3].set_title("Weighted hit rate"); axs[3].set_xlabel("Iter"); axs[3].set_ylabel("p(hit)")
    else:
        axs[3].axis("off")

    for ax in axs:
        ax.grid(alpha=0.3, linewidth=0.6)

    fig2.suptitle("Adaptive parameter evolution", y=0.995)
    fig2.tight_layout()
    if save:
        fig2.savefig(f"{prefix}_param_evolution.png", bbox_inches="tight")