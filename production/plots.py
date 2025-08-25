import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

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