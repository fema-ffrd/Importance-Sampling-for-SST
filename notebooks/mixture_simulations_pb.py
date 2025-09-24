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
from production.metrics import metrics
from production.plots import plot_return_period_summary, plot_two_return_period_summaries


# %% [markdown]
# Load data

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
base_dir = "data/1_interim"
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

# %% [markdown]
# Trinity

# %%
watershed = watersheds["Trinity"]

# %%
#Sample
params = {
    "mu_x_narrow": watershed.watershed_stats["x"],  
    "mu_y_narrow": watershed.watershed_stats["y"],
    "mu_x_wide": watershed.domain_stats["x"],  
    "mu_y_wide": watershed.domain_stats["y"],
    "sd_x_narrow": watershed.watershed_stats["range_x"]*0.2,
    "sd_y_narrow": watershed.watershed_stats["range_y"]*0.2,
    "sd_x_wide": watershed.domain_stats["range_x"],
    "sd_y_wide": watershed.domain_stats["range_y"],
    "mix": 0.8,
    "rho_narrow": -0.7,
    "rho_wide":   0.1,
}

sampler = ImportanceSampler(
    distribution="mixture_trunc_gauss",
    params=params,
    num_simulations=6_000,
    num_realizations=50,
)

mixture_samples = sampler.sample(data = watershed)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
watershed.watershed_gdf.boundary.plot(ax=ax, linewidth=0.8, color='blue', label='Watershed')
plt.scatter(mixture_samples.newx, mixture_samples.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

# %%
mixture_summary = summarize_depths_by_return_period(
    df=mixture_depths,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=2000,
)

# %%
m = metrics(summaries["Trinity"],mixture_summary)
m

# %%
plot_two_return_period_summaries(
    summary1= summaries["Trinity"],          # your first summary DataFrame
    summary2=mixture_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Truncated Normal Sampling",
    title=""
)

# %% [markdown]
# Duwamish

# %%
watershed = watersheds["Duwamish"]

# %%
#Sample
params = {
    "mu_x_narrow": watershed.watershed_stats["x"],  
    "mu_y_narrow": watershed.watershed_stats["y"],
    "mu_x_wide": watershed.domain_stats["x"],  
    "mu_y_wide": watershed.domain_stats["y"],
    "sd_x_narrow": watershed.watershed_stats["range_x"]*0.2,
    "sd_y_narrow": watershed.watershed_stats["range_y"]*0.2,
    "sd_x_wide": watershed.domain_stats["range_x"],
    "sd_y_wide": watershed.domain_stats["range_y"],
    "mix": 0.6,
    "rho_narrow": -0.7,
    "rho_wide":   0.1,
}

sampler = ImportanceSampler(
    distribution="mixture_trunc_gauss",
    params=params,
    num_simulations=6_000,
    num_realizations=50,
)

mixture_samples = sampler.sample(data = watershed)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
watershed.watershed_gdf.boundary.plot(ax=ax, linewidth=0.8, color='blue', label='Watershed')
plt.scatter(mixture_samples.newx, mixture_samples.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

# %%
mixture_summary = summarize_depths_by_return_period(
    df=mixture_depths,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=2000,
)

# %%
m = metrics(summaries["Duwamish"],mixture_summary)
m

# %%
plot_two_return_period_summaries(
    summary1= summaries["Duwamish"],          # your first summary DataFrame
    summary2=mixture_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Truncated Normal Sampling",
    title=""
)

# %% [markdown]
# Kanawha

# %%
watershed = watersheds["Kanawha"]

# %%
#Sample
params = {
    "mu_x_narrow": watershed.watershed_stats["x"],  
    "mu_y_narrow": watershed.watershed_stats["y"],
    "mu_x_wide": watershed.domain_stats["x"],  
    "mu_y_wide": watershed.domain_stats["y"],
    "sd_x_narrow": watershed.watershed_stats["range_x"]*0.2,
    "sd_y_narrow": watershed.watershed_stats["range_y"]*0.3,
    "sd_x_wide": watershed.domain_stats["range_x"],
    "sd_y_wide": watershed.domain_stats["range_y"],
    "mix": 0.8,
    "rho_narrow": -0.1,
    "rho_wide":   0.1,
}

sampler = ImportanceSampler(
    distribution="mixture_trunc_gauss",
    params=params,
    num_simulations=6_000,
    num_realizations=50,
)

mixture_samples = sampler.sample(data = watershed)

# %%
ax = watershed.domain_gdf.boundary.plot(linewidth=0.6, figsize=(9,9))
watershed.watershed_gdf.boundary.plot(ax=ax, linewidth=0.8, color='blue', label='Watershed')
plt.scatter(mixture_samples.newx, mixture_samples.newy, s=0.2, alpha=0.3, rasterized=True)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.show()

# %%
mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

# %%
mixture_summary = summarize_depths_by_return_period(
    df=mixture_depths,       
    precip_col="precip_avg_mm",
    exc_col="exc_prb",
    realization_col="realization",
    k=10.0,                            
    rp_min=2,
    rp_max_cap=2000,
)

# %%
m = metrics(summaries["Kanawha"],mixture_summary)
m

# %%
plot_two_return_period_summaries(
    summary1= summaries["Kanawha"],          # your first summary DataFrame
    summary2=mixture_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Truncated Normal Sampling",
    title=""
)

# %% [markdown]
# Combined

# %%
watersheds

# %%
watershed_name = "Trinity"
watershed = watersheds[watershed_name]

# ----- fixed sampler params -----
params = {
    "mu_x_narrow": float(watershed.watershed_stats["x"]),
    "mu_y_narrow": float(watershed.watershed_stats["y"]),
    "mu_x_wide":   float(watershed.domain_stats["x"]),
    "mu_y_wide":   float(watershed.domain_stats["y"]),
    "sd_x_narrow": float(watershed.watershed_stats["range_x"])*0.2,
    "sd_y_narrow": float(watershed.watershed_stats["range_y"])*0.2,
    "sd_x_wide":   float(watershed.domain_stats["range_x"]),
    "sd_y_wide":   float(watershed.domain_stats["range_y"]),
    "mix": 0.8,
    "rho_narrow": -0.7,
    "rho_wide":   0.1,
}

# where to save depths
out_dir = f"data/2_production/{watershed_name}"
os.makedirs(out_dir, exist_ok=True)

# baseline obs/target summary to compare against
baseline_summary = summaries[watershed_name]

metrics_rows = []

for ns in (1_000,6_000):
    # ---- sample ----
    sampler = ImportanceSampler(
        distribution="mixture_trunc_gauss",
        params=params,
        num_simulations=ns,
        num_realizations=50,   # keep fixed
    )
    mixture_samples = sampler.sample(data=watershed)

    # ---- depths ----
    mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

    # save depths parquet
    depths_path = os.path.join(out_dir, f"mixture_depths_ns{ns}.pq")
    mixture_depths.to_parquet(depths_path, index=False)

    # ---- summarize ----
    mixture_summary = summarize_depths_by_return_period(
        df=mixture_depths,
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,
        rp_min=2,
        rp_max_cap=2000,
    )

    # ---- metrics vs baseline ----
    m = metrics(baseline_summary, mixture_summary)
    m["num_simulations"] = ns
    metrics_rows.append(m)

# final metrics table
metrics_table = pd.concat(metrics_rows, ignore_index=True)
metrics_table
metrics_path = os.path.join(out_dir, f"metrics.pq")
metrics_table.to_parquet(metrics_path, index=False)


# %%
metrics_table

# %%
baseline = 20_000  # baseline number of samples

# Suppose metrics_table has columns:
# 'num_simulations','RMSE_median','RMSE_ci95_low','RMSE_ci95_high'
x_raw = metrics_table["num_simulations"].values

# compute percent reduction relative to baseline
x_pct_reduction = (baseline - x_raw) / baseline * 100.0

rmse_med = metrics_table["rmse_median"].values
rmse_low = metrics_table["rmse_ci95_low"].values
rmse_high = metrics_table["rmse_ci95_high"].values

plt.figure(figsize=(8,5), dpi=150)

# ribbon
plt.fill_between(x_pct_reduction, rmse_low, rmse_high,
                 color="lightblue", alpha=0.3,
                 label="95% CI RMSE")

# median RMSE line
plt.plot(x_pct_reduction, rmse_med, "-o", color="blue", lw=2, label="Median RMSE")

plt.xlabel("Percent reduction from 20k baseline (%)")
plt.ylabel("RMSE")
plt.title("RMSE vs. Percent Reduction in Sample Size")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
watershed_name = "Duwamish"
watershed = watersheds[watershed_name]

# ----- fixed sampler params -----
params = {
    "mu_x_narrow": float(watershed.watershed_stats["x"]),
    "mu_y_narrow": float(watershed.watershed_stats["y"]),
    "mu_x_wide":   float(watershed.domain_stats["x"]),
    "mu_y_wide":   float(watershed.domain_stats["y"]),
    "sd_x_narrow": float(watershed.watershed_stats["range_x"])*0.5,
    "sd_y_narrow": float(watershed.watershed_stats["range_y"])*0.5,
    "sd_x_wide":   float(watershed.domain_stats["range_x"]),
    "sd_y_wide":   float(watershed.domain_stats["range_y"]),
    "mix": 0.8,
    "rho_narrow": -0.7,
    "rho_wide":   0.1,
}

# where to save depths
out_dir = f"data/2_production/{watershed_name}"
os.makedirs(out_dir, exist_ok=True)

# baseline obs/target summary to compare against
baseline_summary = summaries[watershed_name]

metrics_rows = []

for ns in (6_000, 8_000,10_000,12_000,14_000,16_000):
    # ---- sample ----
    sampler = ImportanceSampler(
        distribution="mixture_trunc_gauss",
        params=params,
        num_simulations=ns,
        num_realizations=50,   # keep fixed
    )
    mixture_samples = sampler.sample(data=watershed)

    # ---- depths ----
    mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

    # save depths parquet
    depths_path = os.path.join(out_dir, f"mixture_depths_ns{ns}.pq")
    mixture_depths.to_parquet(depths_path, index=False)

    # ---- summarize ----
    mixture_summary = summarize_depths_by_return_period(
        df=mixture_depths,
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,
        rp_min=2,
        rp_max_cap=2000,
    )

    # ---- metrics vs baseline ----
    m = metrics(baseline_summary, mixture_summary)
    m["num_simulations"] = ns
    metrics_rows.append(m)

# final metrics table
metrics_table = pd.concat(metrics_rows, ignore_index=True)
metrics_table
metrics_path = os.path.join(out_dir, f"metrics.pq")
metrics_table.to_parquet(metrics_path, index=False)

# %%
metrics_table

# %%
metrics_table = pd.read_parquet("data/2_production/Duwamish/metrics.pq")
baseline = 20_000  # baseline number of samples

# Suppose metrics_table has columns:
# 'num_simulations','RMSE_median','RMSE_ci95_low','RMSE_ci95_high'
x_raw = metrics_table["num_simulations"].values

# compute percent reduction relative to baseline
x_pct_reduction = (baseline - x_raw) / baseline * 100.0

rmse_med = metrics_table["rmse_median"].values
rmse_low = metrics_table["rmse_ci95_low"].values
rmse_high = metrics_table["rmse_ci95_high"].values

plt.figure(figsize=(8,5), dpi=150)

# ribbon
plt.fill_between(x_pct_reduction, rmse_low, rmse_high,
                 color="lightblue", alpha=0.3,
                 label="95% CI RMSE")

# median RMSE line
plt.plot(x_pct_reduction, rmse_med, "-o", color="blue", lw=2, label="Median RMSE")

plt.xlabel("Percent reduction from 20k baseline (%)")
plt.ylabel("RMSE")
plt.title("RMSE vs. Percent Reduction in Sample Size")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
watershed_name = "Kanawha"
watershed = watersheds[watershed_name]

# ----- fixed sampler params -----
params = {
    "mu_x_narrow": float(watershed.watershed_stats["x"]),
    "mu_y_narrow": float(watershed.watershed_stats["y"]),
    "mu_x_wide":   float(watershed.domain_stats["x"]),
    "mu_y_wide":   float(watershed.domain_stats["y"]),
    "sd_x_narrow": float(watershed.watershed_stats["range_x"])*0.2,
    "sd_y_narrow": float(watershed.watershed_stats["range_y"])*0.3,
    "sd_x_wide":   float(watershed.domain_stats["range_x"]),
    "sd_y_wide":   float(watershed.domain_stats["range_y"]),
    "mix": 0.8,
    "rho_narrow": -0.1,
    "rho_wide":   0.1,
}

# where to save depths
out_dir = f"data/2_production/{watershed_name}"
os.makedirs(out_dir, exist_ok=True)

# baseline obs/target summary to compare against
baseline_summary = summaries[watershed_name]

metrics_rows = []

for ns in (6_000, 8_000,10_000,12_000,14_000,16_000):
    # ---- sample ----
    sampler = ImportanceSampler(
        distribution="mixture_trunc_gauss",
        params=params,
        num_simulations=ns,
        num_realizations=50,   # keep fixed
    )
    mixture_samples = sampler.sample(data=watershed)

    # ---- depths ----
    mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

    # save depths parquet
    depths_path = os.path.join(out_dir, f"mixture_depths_ns{ns}.pq")
    mixture_depths.to_parquet(depths_path, index=False)

    # ---- summarize ----
    mixture_summary = summarize_depths_by_return_period(
        df=mixture_depths,
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,
        rp_min=2,
        rp_max_cap=2000,
    )

    # ---- metrics vs baseline ----
    m = metrics(baseline_summary, mixture_summary)
    m["num_simulations"] = ns
    metrics_rows.append(m)

# final metrics table
metrics_table = pd.concat(metrics_rows, ignore_index=True)
metrics_table
metrics_path = os.path.join(out_dir, f"metrics.pq")
metrics_table.to_parquet(metrics_path, index=False)

# %%
baseline = 20_000  # baseline number of samples

# Suppose metrics_table has columns:
# 'num_simulations','RMSE_median','RMSE_ci95_low','RMSE_ci95_high'
x_raw = metrics_table["num_simulations"].values

# compute percent reduction relative to baseline
x_pct_reduction = (baseline - x_raw) / baseline * 100.0

rmse_med = metrics_table["rmse_median"].values
rmse_low = metrics_table["rmse_ci95_low"].values
rmse_high = metrics_table["rmse_ci95_high"].values

plt.figure(figsize=(8,5), dpi=150)

# ribbon
plt.fill_between(x_pct_reduction, rmse_low, rmse_high,
                 color="lightblue", alpha=0.3,
                 label="95% CI RMSE")

# median RMSE line
plt.plot(x_pct_reduction, rmse_med, "-o", color="blue", lw=2, label="Median RMSE")

plt.xlabel("Percent reduction from 20k baseline (%)")
plt.ylabel("RMSE")
plt.title("RMSE vs. Percent Reduction in Sample Size")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# Denton

# %%
watershed_name = "Denton"
watershed = watersheds[watershed_name]

# ----- fixed sampler params -----
params = {
    "mu_x_narrow": float(watershed.watershed_stats["x"]),
    "mu_y_narrow": float(watershed.watershed_stats["y"]),
    "mu_x_wide":   float(watershed.domain_stats["x"]),
    "mu_y_wide":   float(watershed.domain_stats["y"]),
    "sd_x_narrow": float(watershed.watershed_stats["range_x"])*1,
    "sd_y_narrow": float(watershed.watershed_stats["range_y"])*1,
    "sd_x_wide":   float(watershed.domain_stats["range_x"]),
    "sd_y_wide":   float(watershed.domain_stats["range_y"]),
    "mix": 0.95,
    "rho_narrow": 0,
    "rho_wide":   0.1,
}

# where to save depths
out_dir = f"data/2_production/{watershed_name}"
os.makedirs(out_dir, exist_ok=True)

# baseline obs/target summary to compare against
baseline_summary = summaries[watershed_name]

metrics_rows = []

for ns in (1_000,6_000):
    # ---- sample ----
    sampler = ImportanceSampler(
        distribution="mixture_trunc_gauss",
        params=params,
        num_simulations=ns,
        num_realizations=50,   # keep fixed
    )
    mixture_samples = sampler.sample(data=watershed)

    # ---- depths ----
    mixture_depths = StormDepthProcessor(watershed).run(mixture_samples, n_jobs=-1)

    # save depths parquet
    depths_path = os.path.join(out_dir, f"mixture_depths_ns{ns}.pq")
    mixture_depths.to_parquet(depths_path, index=False)

    # ---- summarize ----
    mixture_summary = summarize_depths_by_return_period(
        df=mixture_depths,
        precip_col="precip_avg_mm",
        exc_col="exc_prb",
        realization_col="realization",
        k=10.0,
        rp_min=2,
        rp_max_cap=2000,
    )

    # ---- metrics vs baseline ----
    m = metrics(baseline_summary, mixture_summary)
    m["num_simulations"] = ns
    metrics_rows.append(m)

# final metrics table
metrics_table = pd.concat(metrics_rows, ignore_index=True)
metrics_table
metrics_path = os.path.join(out_dir, f"metrics.pq")
metrics_table.to_parquet(metrics_path, index=False)


# %%
watershed_name = "Denton"
watershed = watersheds[watershed_name]

# ----- fixed sampler params -----
params = {
    "mu_x_narrow": float(watershed.watershed_stats["x"]),
    "mu_y_narrow": float(watershed.watershed_stats["y"]),
    "mu_x_wide":   float(watershed.domain_stats["x"]),
    "mu_y_wide":   float(watershed.domain_stats["y"]),
    "sd_x_narrow": float(watershed.watershed_stats["range_x"])*1,
    "sd_y_narrow": float(watershed.watershed_stats["range_y"])*1,
    "sd_x_wide":   float(watershed.domain_stats["range_x"]),
    "sd_y_wide":   float(watershed.domain_stats["range_y"]),
    "mix": 0.95,
    "rho_narrow": 0,
    "rho_wide":   0.1,
}

sampler = ImportanceSampler(
    distribution="mixture_trunc_gauss",
    params=params,
    num_simulations=6000,
    num_realizations=50,   # keep fixed
)
mixture_samples = sampler.sample(data=watershed)


# %%
import matplotlib.pyplot as plt

# Plot domain and watershed boundaries
fig, ax = plt.subplots(figsize=(7,7))
watershed.domain_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
watershed.watershed_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=1, linestyle="--")

# Scatter all samples (x, y)
ax.scatter(
    mixture_samples["newx"].to_numpy(),
    mixture_samples["newy"].to_numpy(),
    s=1, alpha=0.4
)

# Nice axes and title
ax.set_aspect("equal", adjustable="box")
minx, miny, maxx, maxy = watershed.domain_gdf.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Denton — Copula Samples over Domain")
plt.show()

# %%
metrics_table

# %%
plot_two_return_period_summaries(
    summary1= summaries["Denton"],          # your first summary DataFrame
    summary2=mixture_summary,       # your second summary DataFrame
    label1="Uniform Sampling",
    label2="Truncated Normal Sampling",
    title=""
)

# %%
mixture_depths

# %%
duwamish_depths = pd.read_parquet("data/2_production/Duwamish/mixture_depths_ns6000.pq")
kanawha_depths = pd.read_parquet("data/2_production/Kanawha/mixture_depths_ns6000.pq")
trinity_depths = pd.read_parquet("data/2_production/Trinity/mixture_depths_ns6000.pq")
denton_depths = pd.read_parquet("data/2_production/Denton/mixture_depths_ns6000.pq")

duwamish_summary = summarize_depths_by_return_period(df=duwamish_depths,precip_col="precip_avg_mm",exc_col="exc_prb",realization_col="realization",k=10.0,rp_min=2,rp_max_cap=2000,)
kanawha_summary = summarize_depths_by_return_period(df=kanawha_depths,precip_col="precip_avg_mm",exc_col="exc_prb",realization_col="realization",k=10.0,rp_min=2,rp_max_cap=2000,)
trinity_summary = summarize_depths_by_return_period(df=trinity_depths,precip_col="precip_avg_mm",exc_col="exc_prb",realization_col="realization",k=10.0,rp_min=2,rp_max_cap=2000,)
denton_summary = summarize_depths_by_return_period(df=denton_depths,precip_col="precip_avg_mm",exc_col="exc_prb",realization_col="realization",k=10.0,rp_min=2,rp_max_cap=2000,)

#Full Monte Carlo summary
base_dir = "data/1_interim"
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------
# 1) Collect mixture summaries
# ---------------------------
mixture_by_ws = {
    "Duwamish": duwamish_summary.copy(),
    "Kanawha":  kanawha_summary.copy(),
    "Trinity":  trinity_summary.copy(),
    "Denton":   denton_summary.copy(),
}

def ensure_inches(df):
    """
    Ensure the summary has columns: RP, median_in, ci95_low_in, ci95_high_in.
    Converts from *_mm if needed.
    """
    out = df.copy()
    # normalize column names (common possibilities)
    cols = set(out.columns)

    # already in inches?
    need_in = {"RP","median_in","ci95_low_in","ci95_high_in"}
    if need_in <= cols:
        return out.sort_values("RP")

    # try mm -> in
    mm_sets = [
        {"median_mm","ci95_low_mm","ci95_high_mm"},
        {"median","ci95_low","ci95_high"},  # assume these are mm if not labeled
    ]
    for mm_set in mm_sets:
        if mm_set <= cols:
            out["median_in"]    = out[list(mm_set)[0]] / 25.4
            # be explicit to avoid order ambiguity
            out["ci95_low_in"]  = out["ci95_low_mm"]  / 25.4 if "ci95_low_mm"  in cols else out["ci95_low"]  / 25.4
            out["ci95_high_in"] = out["ci95_high_mm"] / 25.4 if "ci95_high_mm" in cols else out["ci95_high"] / 25.4
            return out[["RP","median_in","ci95_low_in","ci95_high_in"]].sort_values("RP")

    raise ValueError("Summary is missing required columns (RP & median/CI in inches or mm).")

# inches-ify mixture
for w in mixture_by_ws:
    mixture_by_ws[w] = ensure_inches(mixture_by_ws[w])

# Full Monte already loaded into `summaries` dict (per your snippet)
fullmonte_by_ws = {w: ensure_inches(summaries[w]) for w in summaries.keys()}

# ---------------------------
# 2) Plot overlay (2x2 grid)
# ---------------------------
watersheds = watershed_names  # keep your existing ordering

x_ticks = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]
x_min, x_max = 2, 2000

fig = plt.figure(figsize=(12, 9), dpi=150)
gs = GridSpec(2, 2, figure=fig, wspace=0.18, hspace=0.14)

# Style: Full Monte (C0), Mixture (C1)
styles = {
    "Full Monte": {"color": "C0", "alpha": 0.22, "linew": 2.2},
    "Mixture":    {"color": "C1", "alpha": 0.22, "linew": 2.2},
}

axes = []
for i, w in enumerate(watersheds):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    axes.append(ax)

    d_full = fullmonte_by_ws[w]
    d_mix  = mixture_by_ws[w]

    # --- ribbons ---
    ax.fill_between(
        d_full["RP"], d_full["ci95_low_in"], d_full["ci95_high_in"],
        alpha=styles["Full Monte"]["alpha"], linewidth=0, color=styles["Full Monte"]["color"]
    )
    ax.fill_between(
        d_mix["RP"], d_mix["ci95_low_in"], d_mix["ci95_high_in"],
        alpha=styles["Mixture"]["alpha"], linewidth=0, color=styles["Mixture"]["color"]
    )

    # --- median lines ---
    ax.plot(d_full["RP"], d_full["median_in"],
            linewidth=styles["Full Monte"]["linew"], color=styles["Full Monte"]["color"])
    ax.plot(d_mix["RP"], d_mix["median_in"],
            linewidth=styles["Mixture"]["linew"], color=styles["Mixture"]["color"])

    # axes formatting
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", which="major", length=4)
    ax.margins(x=0)

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

#fig.subplots_adjust(bottom=0.16)

# global x label
fig.text(0.5, 0.02, "Return period (years)", ha="center", va="center", fontsize=11)

# legend (single, bottom)
legend_elements = [
    Line2D([0], [0], color=styles["Full Monte"]["color"], lw=styles["Full Monte"]["linew"], label="Full Monte"),
    Patch(facecolor=styles["Full Monte"]["color"], alpha=styles["Full Monte"]["alpha"], label="Full Monte 95% CI"),
    Line2D([0], [0], color=styles["Mixture"]["color"], lw=styles["Mixture"]["linew"], label="Mixture (IS)"),
    Patch(facecolor=styles["Mixture"]["color"], alpha=styles["Mixture"]["alpha"], label="Mixture 95% CI"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.8, -0.01),   # slightly above the edge now
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2.8,
)

# save
out_png = os.path.join("data/2_production/plots",
                       "fullmonte_vs_mixture_precip_frequency.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved: {out_png}")


# %%
duwamish_metrics = metrics(summaries["Duwamish"], duwamish_summary)
kanawha_metrics  = metrics(summaries["Kanawha"],  kanawha_summary)
trinity_metrics  = metrics(summaries["Trinity"],  trinity_summary)
denton_metrics   = metrics(summaries["Denton"],   denton_summary)

# %%
# add a column for the watershed name to each metrics DataFrame
duwamish_metrics["watershed"] = "Duwamish"
kanawha_metrics["watershed"]  = "Kanawha"
trinity_metrics["watershed"]  = "Trinity"
denton_metrics["watershed"]   = "Denton"

# combine them into one long DataFrame
all_metrics = pd.concat(
    [duwamish_metrics, kanawha_metrics, trinity_metrics, denton_metrics],
    ignore_index=True
)

# save to CSV
out_csv = "data/2_production/mixture_metrics.csv"
all_metrics.to_csv(out_csv, index=False)

print(f"Saved combined metrics to: {out_csv}")

# %%



