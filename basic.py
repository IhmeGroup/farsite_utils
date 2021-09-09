"""Script for generating basic dataset."""

import os
import numpy as np
import pandas as pd
import datetime as dt

import case
import generate
import raws

n_cases = 3
prototype = case.Case("./data/cases/multi/job.slurm")
fuels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 98, 99]
ensemble_name = "basic"
ensemble_root_dir = "/home/ihme/mbonanni/wildfire_ml/basic/farsite"
ensemble_out_dir = "/home/ihme/mbonanni/wildfire_ml/basic/export"
np.random.seed(42)

shape = (prototype.lcp.num_north, prototype.lcp.num_east)

cases = []
case_ids = []

for i in range(n_cases):
    cases.append(prototype)

    case_ids.append("{0:0{1}d}".format(i, int(np.ceil(np.log10(n_cases-1)))))

    cases[i].name = ensemble_name + "_" + case_ids[i]
    cases[i].root_dir = os.path.join(ensemble_root_dir, case_ids[i])

    cases[i].start_time = dt.datetime(2000, 1, 1, 12, 0)
    cases[i].end_time = dt.datetime(2000, 1, 1, 18, 0)

    # TODO - ignition

    cases[i].lcp.description = "Randomly-generated planar terrain"
    aspect = np.random.uniform(0, 2*np.pi)
    slope = np.random.uniform(0, np.pi/4)
    cases[i].lcp.layers['aspect'].file = ""
    cases[i].lcp.layers['aspect'].unit_opts = np.int16(0) # Degrees
    cases[i].lcp.layers['aspect'].data = np.round(np.zeros(shape) + np.degrees(aspect)).astype(np.int16)
    cases[i].lcp.layers['slope'].file = ""
    cases[i].lcp.layers['slope'].unit_opts = np.int16(0) # Degrees
    cases[i].lcp.layers['slope'].data = np.round(np.zeros(shape) + np.degrees(slope)).astype(np.int16)
    cases[i].lcp.layers['elevation'].file = ""
    cases[i].lcp.layers['elevation'].unit_opts = np.int16(0) # Meters
    cases[i].lcp.layers['elevation'].data = np.round(generate.gradient(shape, aspect, slope)).astype(np.int16)
    cases[i].lcp.layers['fuel'].file = ""
    cases[i].lcp.layers['fuel'].unit_opts = np.int16(0) # No custom and no file
    cases[i].lcp.layers['fuel'].data = generate.randomChoice(shape, fuels, dtype=np.int16)
    cases[i].lcp.layers['cover'].file = ""
    cases[i].lcp.layers['cover'].unit_opts = np.int16(0) # Percent
    cases[i].lcp.layers['cover'].data = generate.randomInteger(shape, 0, 99, dtype=np.int16)

    cases[i].crown_fuels = 21 # on
    cases[i].lcp.layers['height'].file = ""
    cases[i].lcp.layers['height'].unit_opts = np.int16(3) # Meters x 10
    cases[i].lcp.layers['height'].data = generate.randomInteger(shape, 0, 500, dtype=np.int16)
    cases[i].lcp.layers['base'].file = ""
    cases[i].lcp.layers['base'].unit_opts = np.int16(3) # Meters x 10
    cases[i].lcp.layers['base'].data = generate.randomInteger(shape, 0, 100, dtype=np.int16)
    cases[i].lcp.layers['density'].file = ""
    cases[i].lcp.layers['density'].unit_opts = np.int16(3) # kg / m^3
    cases[i].lcp.layers['density'].data = generate.randomInteger(shape, 0, 40, dtype=np.int16)

    cases[i].ground_fuels = 20 # off
    cases[i].lcp.layers['duff'].file = ""
    cases[i].lcp.layers['duff'].data = np.zeros(shape, dtype=np.int16)
    cases[i].lcp.layers['woody'].file = ""
    cases[i].lcp.layers['woody'].data = np.zeros(shape, dtype=np.int16)

    cases[i].weather.elevation = 0
    cases[i].weather.units = raws.Unit.ENGLISH
    cases[i].weather.data['temperature'   ][:] = 77
    cases[i].weather.data['humidity'      ][:] = 25
    cases[i].weather.data['precipitation' ][:] = 0.0
    cases[i].weather.data['wind_speed'    ][:] = np.random.randint(0, 10)
    cases[i].weather.data['wind_direction'][:] = np.random.randint(0, 359)
    cases[i].weather.data['cloud_cover'   ][:] = 25
    
    models = [0] + fuels
    cases[i].fuel_moistures = pd.DataFrame({
        'model':           models,
        '1_hour':          list(np.random.randint(2, 20,  len(models))),
        '10_hour':         list(np.random.randint(2, 20,  len(models))),
        '100_hour':        list(np.random.randint(2, 20,  len(models))),
        'live_herbaceous': list(np.random.randint(2, 100, len(models))),
        'live_woody':      list(np.random.randint(2, 100, len(models)))})
    
    cases[i].spot_probability = 0.0
    cases[i].foliar_moisture_content = 100
    cases[i].crown_fire_method = case.CrownFireMethod.FINNEY
    cases[i].number_processors = 1
    cases[i].out_type = 0

    if not os.path.isdir(cases[i].root_dir):
        os.makedirs(cases[i].root_dir, exist_ok=True)
    cases[i].write()
    cases[i].run()

exported = [False] * n_cases

while not all(exported):
    for i, case in enumerate(cases):
        import code; code.interact(local=locals())
        if case.isDone() and not exported[i]:
            print("Post-processing case " + case_ids[i])
            case.readOutput()
            case.computeBurnMaps()
            case.exportData(os.path.join(ensemble_out_dir, case.name))
            exported[i] = True