# %%
import os
# os.chdir("..")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from SSTImportanceSampling  import Preprocessor, ImportanceSampler, StormDepthProcessor, AdaptParams, AdaptiveMixtureSampler

from production.utils import summarize_depths_by_return_period

# %% [markdown]
# <h3> Preprocess </h3>

# %%
watershed_names = ["Trinity","Kanawha","Duwamish","Denton"]

# %%
#Run and store data
watersheds = {}

for wname in watershed_names:
    print(f"\n=== Processing {wname} ===")
    ws = Preprocessor(
        config_path=f"data/0_source/{wname}/config.json",
        output_folder=f"data/1_interim/{wname}"
    )
    ws.run()
    watersheds[wname] = ws

# %%
#Load data
watersheds = {}

for wname in watershed_names:
    ws = Preprocessor.load(
        config_path=f"data/1_interim/{wname}/config.json"
    )
    watersheds[wname] = ws

# %% [markdown]
# Full Monte Depths

# %%
#Run full Monte Carlo simulations
sampler = ImportanceSampler(
    distribution="uniform",
    params={},
    num_simulations=20_000,
    num_realizations=50
)

for wname, ws in watersheds.items():
    samples = sampler.sample(data=ws)
    depths = StormDepthProcessor(ws).run(samples, n_jobs=-1)
    depths.to_parquet(
        f"data/2_production/{wname}/fullmonte_depths.pq",
        index=False
    )
    print(f"{wname} done.")

# %%
#Load full Monte Carlo simulation results
depths_all = {}
for wname in watersheds.keys():
    path = f"data/2_production/{wname}/fullmonte_depths.pq"
    depths_all[wname] = pd.read_parquet(path)

# %% [markdown]
# Summarize

# %%
for wname, df_depths in depths_all.items():
    if df_depths.empty:
        print(f"Skipping {wname}: no depths")
        continue
    
    summary = summarize_depths_by_return_period(
        df=df_depths,
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,
        rp_min=2,
        rp_max_cap=2000,
    )
    out_path = f"data/2_production/{wname}/fullmonte_summary.pq"
    summary.to_parquet(out_path, index=False)
    print(f"Saved summary for {wname} to {out_path}")

# %% [markdown]
# Plot

# %%
# --- inputs ---
base_dir = "data/2_production"
summary_filename = "fullmonte_summary.pq"

# --- load summaries ---
summaries = {}
for w in watershed_names:
    path = os.path.join(base_dir, w, summary_filename)
    df = pd.read_parquet(path)
    needed = {"RP","median_in","ci95_low_in","ci95_high_in"}
    if not needed <= set(df.columns):
        raise ValueError(f"{w}: summary missing columns {needed - set(df.columns)}")
    summaries[w] = df.sort_values("RP")

# --- plotting ---
x_ticks = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]
x_min, x_max = 2, 2000

fig = plt.figure(figsize=(12, 9), dpi=150)
gs = GridSpec(2, 2, figure=fig, wspace=0.18, hspace=0.14)

axes = []
for i, w in enumerate(watershed_names):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    axes.append(ax)
    d = summaries[w]

    # 95% ribbon
    ax.fill_between(
        d["RP"], d["ci95_low_in"], d["ci95_high_in"],
        alpha=0.25, linewidth=0, color="C0"  # blueish
    )
    # median line
    ax.plot(d["RP"], d["median_in"], linewidth=2.0, color="C0")

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", which="major", length=4)
    ax.margins(x=0)  # no extra space

    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel("Precip (in)")
    ax.set_title(w, fontsize=12, pad=6)

# show x tick labels only on bottom row
for ax in axes[:2]:
    ax.set_xticklabels([])

# remove top/right spines
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# global x label
fig.text(0.5, 0.02, "Return period (years)", ha="center", va="center", fontsize=11)

# --- add one legend at bottom ---
legend_elements = [
    Line2D([0], [0], color="C0", lw=2, label="Median"),
    Patch(facecolor="C0", alpha=0.25, label="95% Confidence Interval")
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=2,
    frameon=False,
    fontsize=11
)

# save high-res PNG
out_png = os.path.join("data/2_production/plots/fullmonte_precip_frequency.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"Saved: {out_png}")



