import pandas as pd
import numpy as np
from tqdm import tqdm
from .precipdepths import compute_depths
from transpose import sample_poisson
from transpose import sample_uniform_centers
from transpose import transpose_storms

def simulate_one_year(df_storms: pd.DataFrame,v_domain_stats: pd.Series,sp_watershed,
                      lam: float,method: str = "uniform",  # "uniform" or "truncnorm"
                      dist_x=None,dist_y=None) -> pd.DataFrame:
    sampled_events = sample_poisson(df_storms, lam)
    if sampled_events.empty:
        return None

    df_transposed = transpose_storms(
        sampled_events,
        v_domain_stats,
        num_simulations=len(sampled_events),
        method=method,
        dist_x=dist_x,
        dist_y=dist_y
    )

    df_depths = compute_depths(df_transposed, sp_watershed)

    if df_depths.empty:
        return None

    return df_depths.loc[[df_depths['depth'].idxmax()]]


def simulate_years(df_storms: pd.DataFrame,v_domain_stats: pd.Series,
                   sp_watershed,lam: float,n_years: int,
                   method: str = "uniform",  # "uniform" or "truncnorm"
                   dist_x=None,dist_y=None) -> pd.DataFrame:
    results = []
    for _ in tqdm(range(n_years)):
        max_event = simulate_one_year(
            df_storms,
            v_domain_stats,
            sp_watershed,
            lam,
            method=method,
            dist_x=dist_x,
            dist_y=dist_y
        )
        if max_event is not None:
            results.append(max_event)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
