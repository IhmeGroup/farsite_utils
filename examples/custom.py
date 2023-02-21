"""Script for generating basic dataset."""

import os
import time
import numpy as np
import pandas as pd
import datetime as dt

from farsite_utils import ensemble
from farsite_utils import case
from farsite_utils import generate
from farsite_utils import raws

def missingCases(batch, out_dir):
    if not os.path.exists(out_dir):
        return [i for i in range(batch.size)]

    file_counts = np.zeros(batch.size, dtype=int)
    for file in os.listdir(out_dir):
        if not os.path.splitext(file)[-1] == '.npy':
            continue
        case_id_number = int(os.path.split(file)[-1].split("_")[1])
        file_counts[case_id_number] += 1

    present_cases = []
    for i in range(batch.size):
        if file_counts[i] == 16:
            present_cases.append(i)
    return [i for i in range(batch.size) if i not in present_cases]

def detectFixes(batch, missing_cases):
    cases_to_run = []
    cases_to_post = []
    for case_id in missing_cases:
        batch.cases[case_id].detectJobID()
        if batch.cases[case_id].ignitionFailed():
            cases_to_run.append(case_id)
        elif batch.cases[case_id].isDone():
            cases_to_post.append(case_id)
    return (cases_to_run, cases_to_post)

seed = 0*42
np.random.seed(seed)
cases_to_run = []
cases_to_post = []
fuels = [101]
print("Loading prototype...")
prototype = case.Case("../prototype/job.slurm")
batch = ensemble.Ensemble(
    name      = "custom",
    root_dir  = "./",
    n_cases   = 10,
    prototype = prototype)
batch.cases_dir_local = "./cases"
batch.out_dir_local = "./export"
verbose = True
stats_only = False
detect_cases_to_fix = False

##########

if cases_to_run or cases_to_post:
    np.random.seed(seed + 1)
if stats_only:
    batch.computeStatistics()
    exit()
elif detect_cases_to_fix:
    np.random.seed(seed + 1)
    print("Detecting cases to fix...")
    missing_cases = missingCases(batch, os.path.join(batch.root_dir, batch.out_dir_local))
    cases_to_run, cases_to_post = detectFixes(batch, missing_cases)
    if cases_to_run:
        print("Building cases: " + ", ".join([str(case) for case in cases_to_run]))
    if cases_to_post:
        print("Post-processing cases: " + ", ".join([str(case) for case in cases_to_post]))
else:
    cases_to_run = [i for i in range(batch.size)]

# Build and write cases that need to be run
if cases_to_run:
    shape = (prototype.lcp.num_north, prototype.lcp.num_east)
    for i in range(batch.size):
        if not (i in cases_to_run):
            continue
        print("Generating case " + batch.caseID(i))

        # Time parameters
        batch.cases[i].start_time = dt.datetime(2000, 1, 1, 8, 0)
        batch.cases[i].end_time = dt.datetime(2000, 1, 4, 1, 0)
        batch.cases[i].timestep = 15

        # Landscape
        batch.cases[i].lcp.description = "Flat terrain"
        batch.cases[i].lcp.latitude = np.int16(0)
        batch.cases[i].lcp.utm_west = 0.0
        batch.cases[i].lcp.utm_east = shape[0] * batch.cases[i].lcp.res_x
        batch.cases[i].lcp.utm_south = 0.0
        batch.cases[i].lcp.utm_north = shape[1] * batch.cases[i].lcp.res_y
        aspect = np.random.uniform(0, 2*np.pi)
        slope = np.random.uniform(0, np.pi/4)
        batch.cases[i].lcp.layers['aspect'].file = ""
        batch.cases[i].lcp.layers['aspect'].unit_opts = np.int16(2) # Azimuth degrees
        batch.cases[i].lcp.layers['aspect'].data = np.round(np.zeros(shape) + np.degrees(aspect)).astype(np.int16)
        batch.cases[i].lcp.layers['slope'].file = ""
        batch.cases[i].lcp.layers['slope'].unit_opts = np.int16(0) # Degrees
        batch.cases[i].lcp.layers['slope'].data = np.round(np.zeros(shape) + np.degrees(slope)).astype(np.int16)
        batch.cases[i].lcp.layers['elevation'].file = ""
        batch.cases[i].lcp.layers['elevation'].unit_opts = np.int16(0) # Meters
        batch.cases[i].lcp.layers['elevation'].data = np.round(generate.gradient(
            shape,
            aspect,
            slope,
            length_scale=batch.cases[i].lcp.res_x)).astype(np.int16)
        batch.cases[i].lcp.layers['fuel'].file = ""
        batch.cases[i].lcp.layers['fuel'].unit_opts = np.int16(0) # No custom and no file
        fuel = np.ones(shape, dtype=np.int16) * 101
        fuel = generate.setLine(fuel, shape[0], shape[1]/2, shape[0]*0.1, shape[1]/2, 99)
        batch.cases[i].lcp.layers['fuel'].data = fuel
        batch.cases[i].lcp.layers['cover'].file = ""
        batch.cases[i].lcp.layers['cover'].unit_opts = np.int16(1) # Percent
        batch.cases[i].lcp.layers['cover'].data = np.ones(shape, dtype=np.int16) * np.random.randint(low=0, high=100, dtype=np.int16)
        batch.cases[i].lcp.crown_fuels = 21 # on
        batch.cases[i].lcp.layers['height'].file = ""
        batch.cases[i].lcp.layers['height'].unit_opts = np.int16(3) # Meters x 10
        batch.cases[i].lcp.layers['height'].data = np.ones(shape, dtype=np.int16) * np.random.randint(low=30, high=500, dtype=np.int16)
        crown_ratio = np.ones(shape) * np.random.uniform(low=0.1, high=1.0)
        batch.cases[i].lcp.layers['base'].file = ""
        batch.cases[i].lcp.layers['base'].unit_opts = np.int16(3) # Meters x 10
        batch.cases[i].lcp.layers['base'].data = np.round((1-crown_ratio) * batch.cases[i].lcp.layers['height'].data).astype(np.int16)
        batch.cases[i].lcp.layers['density'].file = ""
        batch.cases[i].lcp.layers['density'].unit_opts = np.int16(3) # kg / m^3
        batch.cases[i].lcp.layers['density'].data = np.ones(shape, dtype=np.int16) * np.random.randint(low=0, high=40, dtype=np.int16)
        batch.cases[i].lcp.ground_fuels = 20 # off
        batch.cases[i].lcp.layers['duff'].file = ""
        batch.cases[i].lcp.layers['duff'].data = np.zeros(shape, dtype=np.int16)
        batch.cases[i].lcp.layers['woody'].file = ""
        batch.cases[i].lcp.layers['woody'].data = np.zeros(shape, dtype=np.int16)

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
            '1_hour':          list(np.random.randint(2,  40,  len(models))),
            '10_hour':         list(np.random.randint(2,  40,  len(models))),
            '100_hour':        list(np.random.randint(2,  40,  len(models))),
            'live_herbaceous': list(np.random.randint(30, 100, len(models))),
            'live_woody':      list(np.random.randint(30, 100, len(models)))})
        
        # Miscellaneous
        batch.cases[i].spot_probability = 0.0
        batch.cases[i].foliar_moisture_content = 100
        batch.cases[i].crown_fire_method = case.CrownFireMethod.FINNEY
        batch.cases[i].number_processors = 1
        batch.cases[i].out_type = 0

        # Ignition
        width = batch.cases[i].lcp.utm_east - batch.cases[i].lcp.utm_west
        height = batch.cases[i].lcp.utm_north - batch.cases[i].lcp.utm_south
        accessible_fraction = 0.5
        batch.cases[i].ignition.geometry[0] = generate.regularPolygon(
            sides       = 8,
            radius      = 0.01 * width,
            rotation    = 0,
            translation = (
                np.random.uniform(
                    batch.cases[i].lcp.utm_west + (accessible_fraction/2)*width, 
                    batch.cases[i].lcp.utm_east - (accessible_fraction/2)*width),
                np.random.uniform(
                    batch.cases[i].lcp.utm_south + (accessible_fraction/2)*height,
                    batch.cases[i].lcp.utm_north - (accessible_fraction/2)*height)))

    print("Writing cases...")
    batch.write(cases_to_run)
    print("Running cases...")
    batch.run(cases_to_run)

# Read additional cases that need to be post-processed
if cases_to_post:
    print("Reading cases to be post-processed...")
    for case_id in cases_to_post:
        batch.cases[case_id] = case.Case(
            os.path.join(
                batch.cases[case_id].root_dir,
                batch.cases[case_id].jobfile_name_local))
        batch.cases[case_id].detectJobID()
if (cases_to_run + cases_to_post):
    print("Post-processing cases...")
    cases_not_done = batch.postProcess(cases_to_run + cases_to_post, attempts=50, pause_time=60)
else:
    cases_not_done = []
if not cases_not_done:
    print("Computing statistics...")
    batch.computeStatistics()
