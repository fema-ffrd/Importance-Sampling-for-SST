#region Libraries

#%%
import numpy as np
import pandas as pd

import pathlib

from scipy import stats

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.stats.distributions import TruncatedGeneralizedNormal, TruncatedDistribution, MixtureDistribution
from src.stats.distribution_helpers import truncnorm_params

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
#TODO
def read_param_scheme(folder_watershed: str) -> pd.DataFrame:
    path_watershed = pathlib.Path(folder_watershed)

    df_dist_params = pd.read_csv(path_watershed/'scheme/distribution_params.csv')

    # Preprocess parameter scheme
    df_dist_params = \
    (df_dist_params
        .ffill()
        .assign(param_2_name = lambda _: _.param_2_name.replace({'-': ''}))
        .assign(param_2 = lambda _: _.param_2.replace({'-': ''}))
        .assign(_p1 = lambda _: _.param_1.astype(str).str.replace(r'\.0$', '', regex=True))
        .assign(_p2 = lambda _: _.param_2.astype(str).str.replace(r'\.0$', '', regex=True))
        .assign(
            name_file = lambda _: _.acronym + '_' + _.param_1_name + '_' + _._p1 +
            np.where(_.param_2_name == '', '', '_' + _.param_2_name + '_' + _._p2)
        )
        .drop(columns=['_p1', '_p2'])
    )
    
    return df_dist_params

#%%
#TODO
def get_dist_from_scheme(row_dist_params: pd.Series, v_watershed_stats: pd.Series, v_domain_stats: pd.Series) -> tuple:
    match row_dist_params.dist:
        case 'Truncated Normal':
            mult_std = row_dist_params.param_1
    
            print (f'Running for mult_std = {mult_std}')
            dist_x = stats.truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx))
            dist_y = stats.truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy))
        case 'Truncated Generalized Normal':
            beta = row_dist_params.param_1
    
            print (f'Running for beta = {beta}')
            dist_x = TruncatedGeneralizedNormal(
                beta=beta,
                loc=v_watershed_stats.x,
                scale=v_watershed_stats.range_x,
                lower_bound=v_domain_stats.minx,
                upper_bound=v_domain_stats.maxx,
            )
            dist_y = TruncatedGeneralizedNormal(
                beta=beta,
                loc=v_watershed_stats.y,
                scale=v_watershed_stats.range_y,
                lower_bound=v_domain_stats.miny,
                upper_bound=v_domain_stats.maxy,
            )
        case 'Truncated T':
            mult_std = row_dist_params.param_1
            dof = float(row_dist_params.param_2)
    
            print (f'Running for mult_std = {mult_std}, dof = {dof}')
            dist_x = TruncatedDistribution(stats.t(loc=v_watershed_stats.x, scale=v_watershed_stats.range_x*mult_std, df=dof), v_domain_stats.minx, v_domain_stats.maxx)
            dist_y = TruncatedDistribution(stats.t(loc=v_watershed_stats.y, scale=v_watershed_stats.range_y*mult_std, df=dof), v_domain_stats.miny, v_domain_stats.maxy)
        case 'Truncated Normal + Uniform':
            mult_std = float(row_dist_params.param_1)
            w1 = float(row_dist_params.param_2)
    
            print (f'Running for mult_std = {mult_std}, w1 = {w1}')
            dist_x = MixtureDistribution(
                stats.uniform(v_domain_stats.minx, v_domain_stats.range_x),
                stats.truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx)),
                w1
            )
            dist_y = MixtureDistribution(
                stats.uniform(v_domain_stats.miny, v_domain_stats.range_y),
                stats.truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy)),
                w1
            )

    return dist_x, dist_y    

#endregion -----------------------------------------------------------------------------------------
