"""Script for generating real-world dataset."""

import os
import time
import datetime as dt
from enum import Enum
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
import fiona

from farsite_utils import ensemble
from farsite_utils import case
from farsite_utils import generate
from farsite_utils import raws

np.random.seed(42)
cases_to_fix = [95, 124, 250, 256, 391, 583, 710, 790, 858, 861, 892, 914, 924, 962]

if cases_to_fix:
    np.random.seed(43)

n_fuels = 13
fuels = case.FUELS_13
not_burnable_max = 0.3
expected_res = 30.0

data_dir = "/home/ihme/mbonanni/wildfire_ml/data_real"

# Create reader objects
print("Creating reader objects...")
data_raw = {}
data_raw['elevation'] = rasterio.open(os.path.join(data_dir, "LF2016_Elev_200_CONUS/Tif/LC16_Elev_200.tif"))
data_raw['slope']     = rasterio.open(os.path.join(data_dir, "LF2016_Slp_200_CONUS/Tif/LC16_Slp_200.tif"))
data_raw['aspect']    = rasterio.open(os.path.join(data_dir, "LF2016_Asp_200_CONUS/Tif/LC16_Asp_200.tif"))
data_raw['fuel']      = rasterio.open(os.path.join(data_dir, "LF2020_FBFM{0}_200_CONUS/Tif/LC20_F{0}_200.tif".format(n_fuels)))
data_raw['cover']     = rasterio.open(os.path.join(data_dir, "LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif"))
data_raw['height']    = rasterio.open(os.path.join(data_dir, "LF2020_CH_200_CONUS/Tif/LC20_CH_200.tif"))
data_raw['base']      = rasterio.open(os.path.join(data_dir, "LF2020_CBH_200_CONUS/Tif/LC20_CBH_200.tif"))
data_raw['density']   = rasterio.open(os.path.join(data_dir, "LF2020_CBD_200_CONUS/Tif/LC20_CBD_200.tif"))

# Verify resolution
for key in data_raw:
    if data_raw[key].res != (expected_res, expected_res):
        raise ValueError("Unexpected grid resolution: ({0}, {1})".format(*data_raw[key].res))

# Get CA outline
print("Getting CA outline...")
shapefile = fiona.open(os.path.join(data_dir, "cb_2020_us_state_20m/cb_2020_us_state_20m.shp"), "r")
gdf = gpd.GeoDataFrame.from_features([feature for feature in shapefile], crs=shapefile.crs['init'])
columns = list(shapefile.meta["schema"]["properties"]) + ["geometry"]
gdf = gdf[columns]
gdf = gdf.to_crs(epsg=data_raw['fuel'].crs.to_epsg())
CA_shape = gdf[gdf['NAME'] == "California"]
CA_window = rasterio.windows.from_bounds(
    float(CA_shape.bounds.minx),
    float(CA_shape.bounds.miny),
    float(CA_shape.bounds.maxx),
    float(CA_shape.bounds.maxy),
    data_raw['fuel'].transform)

# Read data within CA bounds
print("Reading data...")
data_CA = {}
for key in data_raw:
    data_CA[key] = data_raw[key].read(1, window=CA_window)
    print("Successfully read " + key)

print("Loading prototype...")
prototype = case.Case("../prototype/job.slurm")
batch = ensemble.Ensemble(
    name      = "real",
    root_dir  = "./",
    n_cases   = 1000,
    prototype = case.Case("../prototype/job.slurm"))
batch.cases_dir_local = "./cases"
batch.out_dir_local = "./export"
verbose = True

shape = (prototype.lcp.num_north, prototype.lcp.num_east)

for i in range(batch.size):

    if cases_to_fix and not i in cases_to_fix:
        continue

    print("Generating case " + batch.caseID(i))

    # Time parameters
    batch.cases[i].start_time = dt.datetime(2000, 1, 1, 8, 0)
    batch.cases[i].end_time = dt.datetime(2000, 1, 1, 20, 0)
    batch.cases[i].timestep = 7

    # Select random window
    valid = False
    while not valid:
        rand_x = np.random.randint(0, data_CA['fuel'].shape[0] - shape[0])
        rand_y = np.random.randint(0, data_CA['fuel'].shape[1] - shape[1])
        sample = data_CA['fuel'][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]]
        not_burnable = np.isin(sample, [-9999] + case.FUELS_NB)
        valid = (np.sum(not_burnable) / not_burnable.size) < not_burnable_max

    # Landscape
    batch.cases[i].lcp.description = "Randomly-generated planar terrain"
    batch.cases[i].lcp.crown_fuels = 21 # on
    batch.cases[i].lcp.ground_fuels = 20 # off

    batch.cases[i].lcp.layers['aspect'   ].unit_opts = np.int16(2) # Azimuth degrees
    batch.cases[i].lcp.layers['slope'    ].unit_opts = np.int16(1) # Percent
    batch.cases[i].lcp.layers['elevation'].unit_opts = np.int16(0) # Meters
    batch.cases[i].lcp.layers['fuel'     ].unit_opts = np.int16(0) # No custom and no file
    batch.cases[i].lcp.layers['cover'    ].unit_opts = np.int16(0) # Percent
    batch.cases[i].lcp.layers['height'   ].unit_opts = np.int16(3) # Meters x 10
    batch.cases[i].lcp.layers['base'     ].unit_opts = np.int16(3) # Meters x 10
    batch.cases[i].lcp.layers['density'  ].unit_opts = np.int16(3) # kg / m^3
    batch.cases[i].lcp.layers['duff'     ].unit_opts = np.int16(0) # Mg/ha x 10
    batch.cases[i].lcp.layers['woody'    ].unit_opts = np.int16(0) # Not present

    # DEBUG - compute slope and aspect from elevation
    # elevation = data_CA['elevation'][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    # slope_0, slope_1 = np.gradient(elevation, 30.0)
    # slope_east, slope_north = (slope_1, -slope_0)
    # slope = np.sqrt(slope_east**2 + slope_north**2) * 100
    # aspect = np.degrees(np.arctan2(slope_north, slope_east) - np.pi/2)
    # aspect[aspect < 0] += 360

    # batch.cases[i].lcp.layers['aspect'   ].data = aspect.astype(np.int16)
    # batch.cases[i].lcp.layers['slope'    ].data = slope.astype(np.int16)
    # batch.cases[i].lcp.layers['elevation'].data = elevation.astype(np.int16)
    # DEBUG - compute slope and aspect from elevation

    batch.cases[i].lcp.layers['aspect'   ].data = data_CA['aspect'   ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['slope'    ].data = data_CA['slope'    ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['elevation'].data = data_CA['elevation'][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['fuel'     ].data = data_CA['fuel'     ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['cover'    ].data = data_CA['cover'    ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['height'   ].data = data_CA['height'   ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['base'     ].data = data_CA['base'     ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['density'  ].data = data_CA['density'  ][rand_x:rand_x + shape[0], rand_y:rand_y + shape[1]].astype(np.int16)
    batch.cases[i].lcp.layers['duff'     ].data = np.zeros(shape, dtype=np.int16)
    batch.cases[i].lcp.layers['woody'    ].data = np.zeros(shape, dtype=np.int16)

    # Weather
    batch.cases[i].weather.elevation = 0
    batch.cases[i].weather.units = raws.Unit.METRIC
    batch.cases[i].weather.data = batch.cases[i].weather.data[0:0]
    weather_start_time = dt.datetime(
        year   = batch.cases[i].start_time.year,
        month  = batch.cases[i].start_time.month,
        day    = batch.cases[i].start_time.day,
        hour   = batch.cases[i].start_time.hour - 1,
        minute = 0)
    weather_end_time = dt.datetime(
        year   = batch.cases[i].end_time.year,
        month  = batch.cases[i].end_time.month,
        day    = batch.cases[i].end_time.day,
        hour   = batch.cases[i].end_time.hour + 1,
        minute = 0)
    total_hours = int((weather_end_time - weather_start_time).total_seconds() / 3600)
    wind_speed = np.random.randint(0, 50)
    wind_direction = np.random.randint(0, 359)
    for hours in range(total_hours+1):
        entry = {}
        entry['time'          ] = weather_start_time + dt.timedelta(hours=hours)
        entry['temperature'   ] = 20
        entry['humidity'      ] = 25
        entry['precipitation' ] = 0.0
        entry['wind_speed'    ] = wind_speed
        entry['wind_direction'] = wind_direction
        entry['cloud_cover'   ] = 25
        batch.cases[i].weather.data = batch.cases[i].weather.data.append(entry, ignore_index=True)
    
    # Moisture
    models = [0] + fuels
    batch.cases[i].fuel_moistures = pd.DataFrame({
        'model':           models,
        '1_hour':          list(np.random.randint(2, 40,  len(models))),
        '10_hour':         list(np.random.randint(2, 40,  len(models))),
        '100_hour':        list(np.random.randint(2, 40,  len(models))),
        'live_herbaceous': list(np.random.randint(30, 100, len(models))),
        'live_woody':      list(np.random.randint(30, 100, len(models)))})
    
    # Ignition
    width = batch.cases[i].lcp.utm_east - batch.cases[i].lcp.utm_west
    height = batch.cases[i].lcp.utm_north - batch.cases[i].lcp.utm_south
    accessible_fraction = 0.5
    batch.cases[i].ignition.geometry[0] = generate.regularPolygon(
        sides       = 8,
        radius      = 0.01 * (batch.cases[i].lcp.utm_east - batch.cases[i].lcp.utm_west),
        rotation    = 0,
        translation = (
            np.random.uniform(
                batch.cases[i].lcp.utm_west + (accessible_fraction/2)*width, 
                batch.cases[i].lcp.utm_east - (accessible_fraction/2)*width),
            np.random.uniform(
                batch.cases[i].lcp.utm_south + (accessible_fraction/2)*height,
                batch.cases[i].lcp.utm_north - (accessible_fraction/2)*height)))
    
    # Miscellaneous
    batch.cases[i].spot_probability = 0.0
    batch.cases[i].foliar_moisture_content = 100
    batch.cases[i].crown_fire_method = case.CrownFireMethod.FINNEY
    batch.cases[i].number_processors = 1
    batch.cases[i].out_type = 0

    # Write and run fix cases
    if cases_to_fix:
        batch.cases[i].write()
        batch.cases[i].run()
        while not batch.cases[i].isDone():
            time.sleep(5)
        batch.cases[i].readOutput()
        batch.cases[i].renderOutput(os.path.join(batch.cases[i].root_dir, batch.cases[i].name))
        batch.cases[i].computeBurnMaps()
        batch.cases[i].exportData(os.path.join(batch.root_dir, batch.out_dir_local, batch.cases[i].name))

if not cases_to_fix:
    print("Writing cases...")
    batch.write()
    print("Running cases...")
    batch.run()
    print("Post processing cases...")
    batch.postProcess(attempts=10, pause_time=5)
