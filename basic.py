"""Script for generating basic dataset."""

import os
import copy
import numpy as np
import pandas as pd
import datetime as dt

import case
import generate
from raws import Unit

n_cases = 3
prototype = case.Case("./data/cases/ref_128/job.slurm")
ensemble_name = "basic"
ensemble_root_dir = "/home/ihme/mbonanni/wildfire_ml/basic/farsite"
ensemble_out_dir = "/home/ihme/mbonanni/wildfire_ml/basic/export"
np.random.seed(42)

# LCP 13
# fuels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 98, 99]
# fuels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# LCP 40
fuels_NB = [91, 92, 93, 98, 99]
fuels_GR = [i for i in range(101, 109+1)]
fuels_GS = [i for i in range(121, 124+1)]
fuels_SH = [i for i in range(141, 149+1)]
fuels_TU = [i for i in range(161, 165+1)]
fuels_TL = [i for i in range(181, 189+1)]
fuels_SB = [i for i in range(201, 204+1)]
fuels = fuels_NB + fuels_GR + fuels_GS + fuels_SH + fuels_TU + fuels_TL + fuels_SB

shape = (prototype.lcp.num_north, prototype.lcp.num_east)

cases = []
case_ids = []

for i in range(n_cases):

    # Start with a prototype
    cases.append(copy.deepcopy(prototype))

    # Set paths
    case_ids.append("{0:0{1}d}".format(i, int(np.ceil(np.log10(n_cases-1)))))
    cases[i].name = ensemble_name + "_" + case_ids[i]
    cases[i].root_dir = os.path.join(ensemble_root_dir, case_ids[i])

    # Time parameters
    cases[i].start_time = dt.datetime(2000, 1, 1, 8, 0)
    cases[i].end_time = dt.datetime(2000, 1, 1, 20, 0)
    cases[i].timestep = 7

    # Ignition
    width = cases[i].lcp.utm_east - cases[i].lcp.utm_west
    height = cases[i].lcp.utm_north - cases[i].lcp.utm_south
    accessible_fraction = 0.5
    cases[i].ignition.geometry[0] = generate.regularPolygon(
        sides       = 8,
        radius      = 0.05 * (cases[i].lcp.utm_east - cases[i].lcp.utm_west),
        rotation    = 0,
        translation = (
            np.random.uniform(
                cases[i].lcp.utm_west + (accessible_fraction/2)*width, 
                cases[i].lcp.utm_east - (accessible_fraction/2)*width),
            np.random.uniform(
                cases[i].lcp.utm_south + (accessible_fraction/2)*height,
                cases[i].lcp.utm_north - (accessible_fraction/2)*height)))

    # Landscape
    cases[i].lcp.description = "Randomly-generated planar terrain"
    aspect = np.random.uniform(0, 2*np.pi)
    slope = np.random.uniform(0, np.pi/4)
    # TODO - copy issue
    cases[i].lcp.layers['aspect'].file = ""
    cases[i].lcp.layers['aspect'].unit_opts = np.int16(2) # Azimuth degrees
    cases[i].lcp.layers['aspect'].data = np.round(np.zeros(shape) + np.degrees(aspect)).astype(np.int16)
    cases[i].lcp.layers['slope'].file = ""
    cases[i].lcp.layers['slope'].unit_opts = np.int16(0) # Degrees
    cases[i].lcp.layers['slope'].data = np.round(np.zeros(shape) + np.degrees(slope)).astype(np.int16)
    cases[i].lcp.layers['elevation'].file = ""
    cases[i].lcp.layers['elevation'].unit_opts = np.int16(0) # Meters
    cases[i].lcp.layers['elevation'].data = np.round(generate.gradient(shape, aspect, slope)).astype(np.int16)
    cases[i].lcp.layers['fuel'].file = ""
    cases[i].lcp.layers['fuel'].unit_opts = np.int16(0) # No custom and no file
    fuel_raw = generate.randomPatchy(shape, fuels, 99, 0.9, 4, 10, 3, dtype=np.int16)
    cases[i].lcp.layers['fuel'].data = generate.setBorder(fuel_raw, 5, 99)
    cases[i].lcp.layers['cover'].file = ""
    cases[i].lcp.layers['cover'].unit_opts = np.int16(0) # Percent
    cases[i].lcp.layers['cover'].data = generate.randomInteger(shape, 0, 100, dtype=np.int16)
    cases[i].lcp.crown_fuels = 21 # on
    cases[i].lcp.layers['height'].file = ""
    cases[i].lcp.layers['height'].unit_opts = np.int16(3) # Meters x 10
    cases[i].lcp.layers['height'].data = generate.randomInteger(shape, 30, 500, dtype=np.int16)
    crown_ratio = generate.randomUniform(shape, 0.1, 1.0)
    cases[i].lcp.layers['base'].file = ""
    cases[i].lcp.layers['base'].unit_opts = np.int16(3) # Meters x 10
    cases[i].lcp.layers['base'].data = np.round((1-crown_ratio) * cases[i].lcp.layers['height'].data).astype(np.int16)
    cases[i].lcp.layers['density'].file = ""
    cases[i].lcp.layers['density'].unit_opts = np.int16(3) # kg / m^3
    cases[i].lcp.layers['density'].data = generate.randomInteger(shape, 0, 40, dtype=np.int16)
    cases[i].lcp.ground_fuels = 20 # off
    cases[i].lcp.layers['duff'].file = ""
    cases[i].lcp.layers['duff'].data = np.zeros(shape, dtype=np.int16)
    cases[i].lcp.layers['woody'].file = ""
    cases[i].lcp.layers['woody'].data = np.zeros(shape, dtype=np.int16)

    # Weather
    cases[i].weather.elevation = 0
    cases[i].weather.units = Unit.METRIC
    cases[i].weather.data = cases[i].weather.data[0:0]
    weather_start_time = dt.datetime(
        year   = cases[i].start_time.year,
        month  = cases[i].start_time.month,
        day    = cases[i].start_time.day,
        hour   = cases[i].start_time.hour - 1,
        minute = 0)
    weather_end_time = dt.datetime(
        year   = cases[i].end_time.year,
        month  = cases[i].end_time.month,
        day    = cases[i].end_time.day,
        hour   = cases[i].end_time.hour + 1,
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
        cases[i].weather.data = cases[i].weather.data.append(entry, ignore_index=True)
    
    # Moisture
    models = [0] + fuels
    cases[i].fuel_moistures = pd.DataFrame({
        'model':           models,
        '1_hour':          list(np.random.randint(2, 40,  len(models))),
        '10_hour':         list(np.random.randint(2, 40,  len(models))),
        '100_hour':        list(np.random.randint(2, 40,  len(models))),
        'live_herbaceous': list(np.random.randint(30, 100, len(models))),
        'live_woody':      list(np.random.randint(30, 100, len(models)))})
    
    # Miscellaneous
    cases[i].spot_probability = 0.0
    cases[i].foliar_moisture_content = 100
    cases[i].crown_fire_method = case.CrownFireMethod.FINNEY
    cases[i].number_processors = 1
    cases[i].out_type = 0

    # Write and run case
    if not os.path.isdir(cases[i].root_dir):
        os.makedirs(cases[i].root_dir, exist_ok=True)
    cases[i].write()
    cases[i].run()

exported = [False] * n_cases

# Post-process
while not all(exported):
    for i, case in enumerate(cases):
        import code; code.interact(local=locals())
        if case.isDone() and not exported[i]:
            print("Post-processing case " + case_ids[i])
            case.readOutput()
            case.renderOutput(os.path.join(case.root_dir, case.name))
            case.computeBurnMaps()
            case.exportData(os.path.join(ensemble_out_dir, case.name))
            exported[i] = True