#region Libraries

#%%
import geopandas as gpd
import os
import pathlib

#endregion -----------------------------------------------------------------------------------------
#region Convert geojson to shp

#%%
path_output = pathlib.Path(r'D:\FEMA Innovations\SO3.1\GIS\Shapefiles')

#%%
path_main = pathlib.Path(r'D:\FEMA Innovations\SO3.1\Data')

#%%
v_path_geojson = list(path_main.glob('**/*.geojson'))

#%%
for path_geojson in v_path_geojson:
    # path_geojson = v_path_geojson[0]
    path_shp = path_output/f'{path_geojson.stem}.shp'

    gdf = gpd.read_file(path_geojson)
    gdf.to_file(path_shp)

#endregion -----------------------------------------------------------------------------------------
