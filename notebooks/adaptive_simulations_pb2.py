#region Libraries

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

from SSTImportanceSampling import Preprocessor, ImportanceSampler, StormDepthProcessor, AdaptParams, AdaptiveMixtureSampler, AdaptParams2, AdaptiveMixtureSampler2

from production.utils import summarize_depths_by_return_period
from production.metrics import metrics
from production.plots import plot_return_period_summary, plot_two_return_period_summaries
from production.plots import plot_adaptive_evolution

#endregion -----------------------------------------------------------------------------------------
#region Read Data

# %%
watershed_names = ["Trinity","Kanawha","Duwamish","Denton"]

# %%
#Load data
watersheds = {}

for wname in watershed_names:
    ws = Preprocessor.load(
        config_path=f"data/1_interim/{wname}/config.json"
    )
    watersheds[wname] = ws

# %%
#Full Monte Carlo summary
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

# %%
watershed = watersheds["Trinity"]

#endregion -----------------------------------------------------------------------------------------
#region Variables

#%%
N = 14000
NUM_REALIZATIONS = 50
RP_MAX_CAP = 5000

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling

# %%
params = AdaptParams(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"] * 0.5,
    sd_y_n=watershed.watershed_stats["range_y"] * 0.5,

    mu_x_w=watershed.watershed_stats["x"],
    mu_y_w=watershed.watershed_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"],
    sd_y_w=watershed.domain_stats["range_y"],

    rho_n=-0.7,      # correlation narrow
    rho_w=0.5,       # correlation wide
    mix=0.5,         # initial mixture weight for narrow
)

sampler = AdaptiveMixtureSampler(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed=42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=10, samples_per_iter=5000, depth_threshold=0.0)

# %%
history

# %%
plot_adaptive_evolution(history, watershed.watershed_gdf, watershed.domain_gdf, save=False, prefix="trinity_ais")

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=RP_MAX_CAP,
    use_common_min=True
)

# %%
m = metrics(summaries["Trinity"],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries["Trinity"],          # your first summary DataFrame
    summary2=adaptive_summary,              # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=""
)

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling PB (power)

# %%
params = AdaptParams2(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"] * 0.5,
    sd_y_n=watershed.watershed_stats["range_y"] * 0.5,

    mu_x_w=watershed.watershed_stats["x"],
    mu_y_w=watershed.watershed_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"],
    sd_y_w=watershed.domain_stats["range_y"],

    rho_n=-0.7,      # correlation narrow
    rho_w=0.5,       # correlation wide
    mix=0.5,         # initial mixture weight for narrow

    reward_method='power', # 'threshold', 'power', 'exponential', 'rank', 'hybrid'
)

sampler = AdaptiveMixtureSampler2(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed=42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=10, samples_per_iter=5000, depth_threshold=0.0)

# %%
history

# %%
plot_adaptive_evolution(history, watershed.watershed_gdf, watershed.domain_gdf, save=False, prefix="trinity_ais")

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=RP_MAX_CAP,
    use_common_min=True
)

# %%
m = metrics(summaries["Trinity"],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries["Trinity"],          # your first summary DataFrame
    summary2=adaptive_summary,              # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=""
)

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling PB (exponential)

# %%
params = AdaptParams2(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"] * 0.5,
    sd_y_n=watershed.watershed_stats["range_y"] * 0.5,

    mu_x_w=watershed.watershed_stats["x"],
    mu_y_w=watershed.watershed_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"],
    sd_y_w=watershed.domain_stats["range_y"],

    rho_n=-0.7,      # correlation narrow
    rho_w=0.5,       # correlation wide
    mix=0.5,         # initial mixture weight for narrow

    reward_method='exponential', # 'threshold', 'power', 'exponential', 'rank', 'hybrid'
)

sampler = AdaptiveMixtureSampler2(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed=42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=10, samples_per_iter=5000, depth_threshold=0.0)

# %%
history

# %%
plot_adaptive_evolution(history, watershed.watershed_gdf, watershed.domain_gdf, save=False, prefix="trinity_ais")

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=RP_MAX_CAP,
    use_common_min=True
)

# %%
m = metrics(summaries["Trinity"],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries["Trinity"],          # your first summary DataFrame
    summary2=adaptive_summary,              # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=""
)

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling PB (rank)

# %%
params = AdaptParams2(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"] * 0.5,
    sd_y_n=watershed.watershed_stats["range_y"] * 0.5,

    mu_x_w=watershed.watershed_stats["x"],
    mu_y_w=watershed.watershed_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"],
    sd_y_w=watershed.domain_stats["range_y"],

    rho_n=-0.7,      # correlation narrow
    rho_w=0.5,       # correlation wide
    mix=0.5,         # initial mixture weight for narrow

    reward_method='rank', # 'threshold', 'power', 'exponential', 'rank', 'hybrid'
)

sampler = AdaptiveMixtureSampler2(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed=42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=10, samples_per_iter=5000, depth_threshold=0.0)

# %%
history

# %%
plot_adaptive_evolution(history, watershed.watershed_gdf, watershed.domain_gdf, save=False, prefix="trinity_ais")

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=RP_MAX_CAP,
    use_common_min=True
)

# %%
m = metrics(summaries["Trinity"],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries["Trinity"],          # your first summary DataFrame
    summary2=adaptive_summary,              # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=""
)

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling PB (hybrid)

# %%
params = AdaptParams2(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"] * 0.5,
    sd_y_n=watershed.watershed_stats["range_y"] * 0.5,

    mu_x_w=watershed.watershed_stats["x"],
    mu_y_w=watershed.watershed_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"],
    sd_y_w=watershed.domain_stats["range_y"],

    rho_n=-0.7,      # correlation narrow
    rho_w=0.5,       # correlation wide
    mix=0.5,         # initial mixture weight for narrow

    reward_method='hybrid', # 'threshold', 'power', 'exponential', 'rank', 'hybrid'
)

sampler = AdaptiveMixtureSampler2(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed=42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=10, samples_per_iter=5000, depth_threshold=0.0)

# %%
history

# %%
plot_adaptive_evolution(history, watershed.watershed_gdf, watershed.domain_gdf, save=False, prefix="trinity_ais")

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=RP_MAX_CAP,
    use_common_min=True
)

# %%
m = metrics(summaries["Trinity"],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries["Trinity"],          # your first summary DataFrame
    summary2=adaptive_summary,              # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=""
)

#endregion -----------------------------------------------------------------------------------------

