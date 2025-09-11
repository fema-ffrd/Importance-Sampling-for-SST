import numpy as np
import pandas as pd
def add_return_periods(df: pd.DataFrame,
                      rep_col: str = "rep",
                      precip_col: str = "precip_avg_mm",
                      exc_col: str = "exc_prb",
                      rp_col: str = "return_period",
                      k: float = 10.0) -> pd.DataFrame:
   """
   Sorts `df` within each rep by precipitation (descending)
   and computes return period from exceedance probability.
   Parameters
   ----------
   df : pd.DataFrame
       Input dataframe with at least rep_col, precip_col, exc_col.
   rep_col : str
       Name of column identifying repetition groups.
   precip_col : str
       Column with precipitation values (used for sorting).
   exc_col : str
       Column with exceedance probability values.
   rp_col : str
       Name of the new return period column.
   k : float
       Rate constant used in formula.
   Returns
   -------
   pd.DataFrame
       Copy of df with an added return_period column.
   """
   df = (
       df
       .sort_values([rep_col, precip_col], ascending=[True, False])
       .reset_index(drop=True)
       .copy()
   )
   df[rp_col] = 1.0 / (1.0 - np.exp(-k * df[exc_col].astype(float)))
   return df