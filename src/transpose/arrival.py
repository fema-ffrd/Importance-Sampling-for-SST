import numpy as np

def sample_poisson(df_storms, lam = 10, random_state=None):
    np.random.seed(random_state)
    k = np.random.poisson(lam)
    k = min(k, len(df_storms))
    return df_storms.sample(n=k, replace=False, random_state=random_state)