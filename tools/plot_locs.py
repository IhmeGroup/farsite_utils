import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
import fiona
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.font_manager

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

_XSMALL_SIZE = 12
_SMALL_SIZE = 14
_MEDIUM_SIZE = 16
_BIGGER_SIZE = 18

plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=_SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=_XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title

home_dir = "/usr/workspace/bonanni1/wildfire_ml/real"
data_dir = "../data_real"
map_file = "map.npy"
map_scale = 0.1
locs_file = "real_00/locs_1.csv"

os.chdir(home_dir)

if os.path.exists(map_file):
    print("Reading data...")
    map_data = np.load(map_file)
    print("Successfully read map data")
else:
    # Create reader objects
    print("Creating reader objects...")
    map_reader = rasterio.open(os.path.join(data_dir, "LF2016_Asp_200_CONUS/Tif/LC16_Asp_200.tif"))

    # Get CA outline
    print("Getting CA outline...")
    shapefile = fiona.open(os.path.join(data_dir, "cb_2020_us_state_20m/cb_2020_us_state_20m.shp"), "r")
    gdf = gpd.GeoDataFrame.from_features([feature for feature in shapefile], crs=shapefile.crs['init'])
    columns = list(shapefile.meta["schema"]["properties"]) + ["geometry"]
    gdf = gdf[columns]
    gdf = gdf.to_crs(epsg=map_reader.crs.to_epsg())
    CA_shape = gdf[gdf['NAME'] == "California"]
    CA_window = rasterio.windows.from_bounds(
        float(CA_shape.bounds.minx),
        float(CA_shape.bounds.miny),
        float(CA_shape.bounds.maxx),
        float(CA_shape.bounds.maxy),
        map_reader.transform)

    # Read data within CA bounds
    print("Reading data...")
    map_data = map_reader.read(1, window=CA_window)

    # Process data
    print("Processing map data...")
    map_data = map_data.astype(np.float32)
    map_data[np.isclose(map_data, -9999)] = np.nan
    map_data = rescale(map_data, map_scale, anti_aliasing=False)

    # Write data
    print("Writing data...")
    np.save("map.npy", map_data)

locs = pd.read_csv(locs_file, index_col=0)
locs_scaled = locs * map_scale

plt.imshow(map_data)
plt.scatter(locs_scaled['y'], locs_scaled['x'], s=20, c='k', marker='x')
plt.axis('off')
plt.savefig("map.png", bbox_inches='tight', dpi=300)
plt.show()
