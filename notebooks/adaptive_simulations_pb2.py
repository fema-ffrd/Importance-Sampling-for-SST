#region Library

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

from SSTImportanceSampling  import Preprocessor, ImportanceSampler, StormDepthProcessor, AdaptParams, AdaptiveMixtureSampler, AdaptParams2, AdaptiveMixtureSampler2

from production.utils import summarize_depths_by_return_period
from production.metrics import metrics
from production.plots import plot_return_period_summary, plot_two_return_period_summaries
from production.plots import plot_adaptive_evolution

import pickle

#endregion -----------------------------------------------------------------------------------------
#region Read Data

# %%
# watershed_names = ["Trinity","Kanawha","Duwamish","Denton"]
watershed_names = ["Trinity","Kanawha","Duwamish"]

# %%
#Load data
watersheds = {}

for wname in watershed_names:
    ws = Preprocessor.load(
        config_path=f"data/1_interim/{wname}/config.json"
    )
    watersheds[wname] = ws

# %% Results location
base_dir = "data/2_production"

# %%
#Full Monte Carlo summary
summary_filename = "fullmonte_summary.pq"

summaries_uniform = {}
for w in watershed_names:
    path = os.path.join(base_dir, w, summary_filename)
    df = pd.read_parquet(path)
    needed = {"RP","median_in","ci95_low_in","ci95_high_in"}
    if not needed <= set(df.columns):
        raise ValueError(f"{w}: summary missing columns {needed - set(df.columns)}")
    summaries_uniform[w] = df.sort_values("RP")

# %%
#Full Monte Carlo depths
results_filename = "fullmonte_depths.pq"

depths_uniform = {}
for w in watershed_names:
    path = os.path.join(base_dir, w, results_filename)
    path = f"data/2_production/{w}/fullmonte_depths.pq"
    depths_uniform[w] = pd.read_parquet(path)

# %%
#True summary
summary_filename = "true_summary.pq"

summaries_true = {}
for w in watershed_names:
    path = os.path.join(base_dir, w, summary_filename)
    df = pd.read_parquet(path)
    needed = {"RP","median_in","ci95_low_in","ci95_high_in"}
    if not needed <= set(df.columns):
        raise ValueError(f"{w}: summary missing columns {needed - set(df.columns)}")
    summaries_true[w] = df.sort_values("RP")

# %%
#True depths
results_filename = "true_depths.pq"

depths_true = {}
for w in watershed_names:
    path = os.path.join(base_dir, w, results_filename)
    path = f"data/2_production/{w}/fullmonte_depths.pq"
    depths_true[w] = pd.read_parquet(path)

#endregion -----------------------------------------------------------------------------------------
#region Variables

# %%
watershed_name = "Duwamish"
watershed_name = "Trinity"
watershed_name = "Kanawha"
# watershed_name = "Denton"

# %%
watershed = watersheds[watershed_name]

# %%
ADAPT_NUM_ITER = 10
ADAPT_SAM_PER_ITER = 1_000

# %%
N = 4_000
NUM_REALIZATIONS = 50

# %%
# RP_MAX_CAP = 5000

# %%
folder_results = 'Temp'

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling

# %%
params = AdaptParams(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"],
    sd_y_n=watershed.watershed_stats["range_y"],
    rho_n=0,        # correlation narrow

    mu_x_w=watershed.domain_stats["x"],
    mu_y_w=watershed.domain_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"]/np.sqrt(12),
    sd_y_w=watershed.domain_stats["range_y"]/np.sqrt(12),
    rho_w=0,        # correlation wide

    mix=0.8,        # initial mixture weight for narrow

    alpha = 0.75,
)

sampler = AdaptiveMixtureSampler(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed = 42
)

# %%
# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=ADAPT_NUM_ITER, samples_per_iter=ADAPT_SAM_PER_ITER)

# %%
history

# %%
plot_adaptive_evolution(
    history, 
    watershed_gdf=watershed.watershed_gdf, 
    domain_gdf=watershed.domain_gdf, 
    save=False, 
    prefix=f"{watershed_name}_ais",
)

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
final_df

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=False)
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
    rp_max_cap=2000,
    use_common_min=True,
)

# %%
m = metrics(summaries_uniform[watershed_name],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries_uniform[watershed_name],          # your first summary DataFrame
    summary2=adaptive_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=f'{watershed_name}',
    save=False
)

# %%
with open(f'{folder_results}/{watershed_name}_params_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(params, f)
with open(f'{folder_results}/{watershed_name}_sampler_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(sampler, f)
with open(f'{folder_results}/{watershed_name}_history_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(history, f)
with open(f'{folder_results}/{watershed_name}_final_df_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(final_df, f)

#endregion -----------------------------------------------------------------------------------------
#region Adaptive Sampling (Storm-weighted)

# %%
params = AdaptParams2(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"],
    sd_y_n=watershed.watershed_stats["range_y"],
    rho_n=0,        # correlation narrow

    mu_x_w=watershed.domain_stats["x"],
    mu_y_w=watershed.domain_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"]/np.sqrt(12),
    sd_y_w=watershed.domain_stats["range_y"]/np.sqrt(12),
    rho_w=0,        # correlation wide

    mix=0.8,        # initial mixture weight for narrow

    alpha = 0.75,
)

sampler = AdaptiveMixtureSampler2(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed = 42
)

# %%
# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=ADAPT_NUM_ITER, samples_per_iter=ADAPT_SAM_PER_ITER)

# # %%
# self = sampler
# seed = 42
# n=N
# num_realizations=NUM_REALIZATIONS
# with_depths=True

# %%
history

# %%
plot_adaptive_evolution(
    history, 
    watershed_gdf=watershed.watershed_gdf, 
    domain_gdf=watershed.domain_gdf, 
    save=False, 
    prefix=f"{watershed_name}_ais",
)

# %%
final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

# %%
final_df

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=False)
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
    rp_max_cap=2000,
    use_common_min=True,
)

# %%
m = metrics(summaries_uniform[watershed_name],adaptive_summary)
m

# %%
adaptive_summary

# %%
plot_two_return_period_summaries(
    summary1=summaries_uniform[watershed_name],          # your first summary DataFrame
    summary2=adaptive_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=f'{watershed_name}',
    save=False
)

# %%
with open(f'{folder_results}/{watershed_name}_params_2_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(params, f)
with open(f'{folder_results}/{watershed_name}_sampler_2_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(sampler, f)
with open(f'{folder_results}/{watershed_name}_history_2_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(history, f)
with open(f'{folder_results}/{watershed_name}_final_df_2_{N}x{NUM_REALIZATIONS}.pkl', 'wb') as f:
    pickle.dump(final_df, f)

#endregion -----------------------------------------------------------------------------------------
#region rho Sensitivity

# %%
params = AdaptParams(
    mu_x_n=watershed.watershed_stats["x"],
    mu_y_n=watershed.watershed_stats["y"],
    sd_x_n=watershed.watershed_stats["range_x"],
    sd_y_n=watershed.watershed_stats["range_y"],
    rho_n=0,        # correlation narrow

    mu_x_w=watershed.domain_stats["x"],
    mu_y_w=watershed.domain_stats["y"],
    sd_x_w=watershed.domain_stats["range_x"]/np.sqrt(12),
    sd_y_w=watershed.domain_stats["range_y"]/np.sqrt(12),
    rho_w=0,        # correlation wide

    mix=0.8,        # initial mixture weight for narrow

    alpha = 0.75,
)

sampler = AdaptiveMixtureSampler(
    data=watershed,                   
    params=params,
    precip_cube=watershed.cumulative_precip,
    seed = 42
)

# Adapt does NOT take data or seed
history = sampler.adapt(num_iterations=ADAPT_NUM_ITER, samples_per_iter=ADAPT_SAM_PER_ITER)

for rho_n in np.arange(0.05, 1.0, 0.05):
    print (f'rho={rho_n:.2f}')

    # 
    final_df = sampler.sample_final(n=N, num_realizations=NUM_REALIZATIONS, with_depths=True)

    # # 
    # final_df

    # # 
    # ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
    # plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=False)
    # plt.gca().set_aspect("equal")
    # plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    # plt.show()

    # 
    adaptive_summary = summarize_depths_by_return_period(
        df=final_df,       
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,                            
        rp_min=2,
        rp_max_cap=2000,
        use_common_min=True,
    )

    # # %%
    # m = metrics(summaries[watershed_name],adaptive_summary)
    # m

    # # 
    # adaptive_summary

    # 
    plot_two_return_period_summaries(
        summary1= summaries_uniform[watershed_name],          # your first summary DataFrame
        summary2=adaptive_summary,       # your second summary DataFrame
        label1="Uniform Sampling",
        label2="Adaptive Sampling",
        title=f'{watershed_name} rho={rho_n:.2f}',
        save=True
    )


#endregion -----------------------------------------------------------------------------------------
#region Read Data

# %%
with open(f'{folder_results}/{watershed_name}_params_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    params = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_sampler_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    sampler = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_history_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    history = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_final_df_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    final_df = pickle.load(f)

# %%
with open(f'{folder_results}/{watershed_name}_params_2_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    params = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_sampler_2_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    sampler = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_history_2_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    history = pickle.load(f)
with open(f'{folder_results}/{watershed_name}_final_df_2_{N}x{NUM_REALIZATIONS}.pkl', 'rb') as f:
    final_df = pickle.load(f)

#endregion -----------------------------------------------------------------------------------------
#region Plots

# %%
adaptive_summary = summarize_depths_by_return_period(
    df=final_df,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=2000,
    use_common_min=True,
)

# %%
plot_adaptive_evolution(
    history, 
    watershed_gdf=watershed.watershed_gdf, 
    domain_gdf=watershed.domain_gdf, 
    save=False, 
    prefix=f"{watershed_name}_ais",
)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
plt.scatter(final_df.newx, final_df.newy, s=0.2, alpha=0.3, rasterized=False)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
plot_two_return_period_summaries(
    summary1=summaries_uniform[watershed_name],          # your first summary DataFrame
    summary2=adaptive_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Adaptive Sampling",
    title=f'{watershed_name}',
    save=False
)

# %%
plot_two_return_period_summaries(
    summary1=summaries_true[watershed_name],          # your first summary DataFrame
    summary2=adaptive_summary,       # your second summary DataFrame
    label1="Ground Truth",
    label2="Adaptive Sampling",
    title=f'{watershed_name}',
    save=False
)

# %%
plot_two_return_period_summaries(
    summary1=summaries_true[watershed_name],          # your first summary DataFrame
    summary2=summaries_uniform[watershed_name],       # your second summary DataFrame
    label1="Ground Truth",
    label2="Uniform Sampling",
    title=f'{watershed_name}',
    save=False
)

#endregion -----------------------------------------------------------------------------------------
