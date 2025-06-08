#region Libraries

#%%
import geopandas as gpd

import xarray as xr

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def shift_gdf(gdf: gpd.GeoDataFrame, x_del: float, y_del: float) -> gpd.GeoDataFrame:
    '''Shift a geodataframe in x and y directions.

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe to shift.
        x_del (float): Shift in x direction.
        y_del (float): Shift in y direction.

    Returns:
        gpd.GeoDataFrame: Shifted geodataframe.
    '''
    gdf_shifted = gpd.GeoDataFrame(geometry=gdf.geometry.translate(x_del, y_del))

    return gdf_shifted

#%%
def shift_storm(xr_dataset: xr.Dataset, x_del: float, y_del: float) -> xr.Dataset:
    '''Shift a storm raster in x and y directions.

    Args:
        xr_dataset (xr.Dataset): xarray dataset of storm raster.
        x_del (float): Shift in x direction.
        y_del (float): Shift in y direction.

    Returns:
        xr.Dataset: Shifted xarray dataset of storm raster.
    '''
    xr_dataset = xr_dataset.copy()
    xr_dataset_shifted = xr_dataset.assign_coords({
        "x": xr_dataset.x + x_del,
        "y": xr_dataset.y + y_del
    })
    return xr_dataset_shifted

#endregion -----------------------------------------------------------------------------------------
#region Tests

# #%%
# sp_watershed=gpd.GeoDataFrame()
# df_storm_sample=pd.DataFrame()

# #%%
# sp_watershed.to_file('temp.shp')

# #%%
# _df_storm_sample = df_storm_sample.iloc[1]
# path_storm = _df_storm_sample.path
# x_del = _df_storm_sample.x_del
# y_del = _df_storm_sample.y_del

# #%%
# sp_watershed_shifted = shift_gdf(sp_watershed, x_del, y_del)

# #%%
# sp_watershed_shifted.to_file('temp_shifted.shp')

# #%%
# v_x_del = df_storm_sample.x_del
# v_y_del = df_storm_sample.y_del

# shift = [(x,y) for x,y in zip(v_x_del, v_y_del)]

# gpd.GeoSeries(np.repeat(sp_watershed.geometry, len(shift))).transform(lambda _: _ + shift)


# #%%
# df_xy = gpd.GeoSeries(np.repeat(sp_watershed.geometry, len(shift))).reset_index().get_coordinates().reset_index()
# df_del = pd.DataFrame(dict(x_del = v_x_del, y_del = v_y_del)).reset_index()

# df_xy = \
# (df_xy
#     .merge(df_del, on='index')
# )

# #%%
# df_xy = gpd.GeoSeries(np.repeat(sp_watershed.geometry, len(shift))).reset_index().get_coordinates()
# df_del = pd.DataFrame(dict(x_del = v_x_del, y_del = v_y_del))

# (df_xy
#     .merge(df_del)
# )

# #%%
# sp_centroid = gpd.GeoDataFrame(geometry=gpd.points_from_xy([_df_storm_sample.x], [_df_storm_sample.y]))
# sp_centroid_new = gpd.GeoDataFrame(geometry=gpd.points_from_xy([_df_storm_sample.x_sampled], [_df_storm_sample.y_sampled]))
# # sp_centroid_new_all = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df_storm_sample.x_sampled, df_storm_sample.y_sampled))

# sp_centroid.to_file('temp_centroid.shp')
# sp_centroid_new.to_file('temp_centroid_new.shp')
# # sp_centroid_new_all.to_file('temp_centroid_new_all.shp')


# #%%
# import xarray as xr
# import rioxarray as rxr

# #%%
# xr_storm = xr.open_dataset(path_storm)

# #%%
# for i in range(1000):
#     xr_storm_shifted = shift_storm(xr_storm, x_del, y_del)

# #%%
# xr_storm_shifted['band_data'].rio.to_raster('temp_shifted.tif')

#endregion -----------------------------------------------------------------------------------------
