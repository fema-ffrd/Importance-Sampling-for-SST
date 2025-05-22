import pandas as pd
import numpy as np

def get_df_freq_curve(depths, probs):
    # Table of depths and probabilities
    df_prob_mc = pd.DataFrame(dict(
        depth = depths,
        prob = probs
    ))
    
    # Exceedence probability
    df_prob_mc = \
    (df_prob_mc
        .sort_values('depth', ascending=False)
        .assign(prob_exceed = lambda _: _.prob.cumsum())
        .assign(return_period = lambda _: 1/_.prob_exceed)
    )

    return df_prob_mc    


def get_return_period(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values('depth', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    df_sorted['prob_exceed'] = (np.arange(1, n + 1)) / (n + 1)
    df_sorted['return_period'] = 1 / df_sorted['prob_exceed']
    return df_sorted


def get_return_period_poisson(depths, probs, lambda_rate=10):
    df_prob_mc = pd.DataFrame({
        'depth': depths,
        'prob': probs
    })

    df_prob_mc = (
        df_prob_mc
        .sort_values('depth', ascending=False)
        .assign(prob_exceed=lambda _: _.prob.cumsum())
        .assign(prob_exceed_poisson=lambda _: 1 - np.exp(-lambda_rate * _.prob_exceed))
        .assign(return_period=lambda _: 1 / _.prob_exceed_poisson)
    )

    return df_prob_mc