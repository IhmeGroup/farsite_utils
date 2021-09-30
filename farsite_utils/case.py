"""This module contains utilities for managing FARSITE cases."""

import os
import subprocess
import warnings
from enum import Enum
import datetime as dt
import pandas as pd
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union
from shapely.prepared import prep
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager

from . import landscape
from . import raws
from . import sbatch
from . import ascii_data

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

_DEFAULT_YEAR = 2000
_BURN_PERIODS_COLS = ['start', 'end']
_FUEL_MOISTURES_COLS = ['model', '1_hour', '10_hour', '100_hour', 'live_herbaceous', 'live_woody']

FUELS_NB = [91, 92, 93, 98, 99]
FUELS_RO = [i for i in range(  1,  13+1)]
FUELS_GR = [i for i in range(101, 109+1)]
FUELS_GS = [i for i in range(121, 124+1)]
FUELS_SH = [i for i in range(141, 149+1)]
FUELS_TU = [i for i in range(161, 165+1)]
FUELS_TL = [i for i in range(181, 189+1)]
FUELS_SB = [i for i in range(201, 204+1)]
FUELS_13_BURNABLE = FUELS_RO
FUELS_40_BURNABLE = FUELS_GR + FUELS_GS + FUELS_SH + FUELS_TU + FUELS_TL + FUELS_SB
FUELS_13 = FUELS_NB + FUELS_13_BURNABLE
FUELS_40 = FUELS_NB + FUELS_40_BURNABLE


class CrownFireMethod(Enum):
    FINNEY = 1
    REINHARDT = 2


class LineType(Enum):
    TITLE = 1
    NAME_VALUE = 2
    EMPTY = 3


class Case:
    """This class represents an individual FARSITE case."""

    def __init__(self, jobfile_name=None):
        # Inputs
        self.sbatch = sbatch.SBatch()
        self.name = "case"
        self.root_dir = "./"
        self.out_dir_local = "output/"
        self.landscape_dir_local = "landscape/"
        self.ignition_dir_local = "ignition/"
        self.jobfile_name_local = "job.slurm"
        self.start_time = dt.datetime(_DEFAULT_YEAR, 1, 1)
        self.end_time = dt.datetime(_DEFAULT_YEAR, 1, 1)
        self.burn_periods_count = 0
        self.burn_periods = pd.DataFrame(columns = _BURN_PERIODS_COLS)
        self.timestep = 60
        self.distance_res = 30.0
        self.perimeter_res = 60.0
        self.min_ignition_vertex_distance = 15.0
        self.spot_grid_resolution = 15.0
        self.spot_probability = 0.05
        self.spot_ignition_delay = 0
        self.minimum_spot_distance = 30
        self.spotting_seed = 0
        self.acceleration_on = True
        self.fuel_moistures_count = 0
        self.fuel_moistures = pd.DataFrame(columns = _FUEL_MOISTURES_COLS)
        self.weather = raws.RAWS()
        self.foliar_moisture_content = 100
        self.crown_fire_method = CrownFireMethod.FINNEY
        self.number_processors = 1
        self.lcp = landscape.Landscape()
        self.ignition = gpd.GeoDataFrame()
        self.out_type = 0
        self.job_id = None

        # Outputs
        self.arrival_time       = ascii_data.ASCIIData()
        self.crown_fire         = ascii_data.ASCIIData()
        self.flame_length       = ascii_data.ASCIIData()
        self.heat_per_unit_area = ascii_data.ASCIIData()
        self.ignitions          = ascii_data.ASCIIData()
        self.intensity          = ascii_data.ASCIIData()
        self.reaction_intensity = ascii_data.ASCIIData()
        self.spot_grid          = ascii_data.ASCIIData()
        self.spread_direction   = ascii_data.ASCIIData()
        self.spread_rate        = ascii_data.ASCIIData()
        self.spots = gpd.GeoDataFrame()
        self.perimeters = gpd.GeoDataFrame()
        self.perimeters_merged = gpd.GeoDataFrame()
        self.burn = None
        self.verbose = False

        if jobfile_name:
            self.read(jobfile_name)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    @property
    def name(self):
        """Name for this case."""
        return self._name
    

    @name.setter
    def name(self, value):
        """Set name, adjusting slurm option appropriately."""
        if not isinstance(value, str):
            raise TypeError("Case.name must be a string")
        self._name = value
        self.sbatch.set_option("-J", value)
    

    @property
    def burn_periods(self):
        """Table of burn periods."""
        return self._burn_periods
    

    @burn_periods.setter
    def burn_periods(self, value):
        """Set table of burn periods, adjusting count appropriately."""
        if not (Counter(value.columns) == Counter(_BURN_PERIODS_COLS)):
            raise ValueError(
                "Data must have the following columns: " + ", ".join(_BURN_PERIODS_COLS))
        self._burn_periods = value
        self.burn_periods_count = len(value.index)
    

    @property
    def fuel_moistures(self):
        """Table of fuel moistures data."""
        return self._fuel_moistures
    

    @fuel_moistures.setter
    def fuel_moistures(self, value):
        """Set table of fuel moistures data, adjusting count appropriately."""
        if not (Counter(value.columns) == Counter(_FUEL_MOISTURES_COLS)):
            raise ValueError(
                "Data must have the following columns: " + ", ".join(_FUEL_MOISTURES_COLS))
        self._fuel_moistures = value
        self.fuel_moistures_count = len(value.index)
    

    def read(self, jobfile_name):
        """Read full case by following paths from slurm script."""
        [self.root_dir, self.jobfile_name_local] = os.path.split(jobfile_name)
        self.sbatch.read(jobfile_name)
        runfile_name = os.path.join(self.root_dir, self.sbatch.runfile_name_local)
        with open(runfile_name, "r") as file:
            line = file.readline()
        args = line.split(" ")
        self.lcp.read(os.path.join(self.root_dir, os.path.splitext(args[0])[0]))
        self.readInput(os.path.join(self.root_dir, args[1]))
        self.ignition = gpd.read_file(os.path.join(self.root_dir, args[2]))
        out_prefix_local = args[4]
        [self.out_dir_local, name] = os.path.split(out_prefix_local)
        self.name = name
        self.out_type = int(args[5])
    

    def __writeInputHeader(self, file):
        """Write the header of the input file."""
        file.write("FARSITE INPUTS FILE VERSION 1.0\n")
        file.write("FARSITE_START_TIME: {0:02d} {1:02d} {2:02d}{3:02d}\n".format(
            self.start_time.month,
            self.start_time.day,
            self.start_time.hour,
            self.start_time.minute))
        file.write("FARSITE_END_TIME: {0:02d} {1:02d} {2:02d}{3:02d}\n".format(
            self.end_time.month,
            self.end_time.day,
            self.end_time.hour,
            self.end_time.minute))
        file.write("FARSITE_TIMESTEP: {0:d}\n".format(self.timestep))
        file.write("FARSITE_DISTANCE_RES: {0:.1f}\n".format(self.distance_res))
        file.write("FARSITE_PERIMETER_RES: {0:.1f}\n".format(self.perimeter_res))
        file.write("FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {0:.1f}\n".format(self.min_ignition_vertex_distance))
        file.write("NUMBER_PROCESSORS: {0:d}\n".format(self.number_processors))


    def __writeInputBurnPeriods(self, file):
        """Write the table of burn periods to the input file."""
        file.write("FARSITE_BURN_PERIODS: {0:d}\n".format(self.burn_periods_count))
        for i in range(self.burn_periods_count):
            file.write("{0:02d} {1:02d} {2:02d}{3:02d} {4:02d}{5:02d}\n".format(
                self.burn_periods.loc[i]['start'].month,
                self.burn_periods.loc[i]['start'].day,
                self.burn_periods.loc[i]['start'].hour,
                self.burn_periods.loc[i]['start'].minute,
                self.burn_periods.loc[i]['end'].hour,
                self.burn_periods.loc[i]['end'].minute))


    def __writeInputFuelMoistures(self, file):
        """Write the table of fuel moistures data to the input file."""
        file.write("FUEL_MOISTURES_DATA: {0:d}\n".format(self.fuel_moistures_count))
        for i in range(self.fuel_moistures_count):
            file.write("{0} {1} {2} {3} {4} {5}\n".format(
                self.fuel_moistures.loc[i]['model'],
                self.fuel_moistures.loc[i]['1_hour'],
                self.fuel_moistures.loc[i]['10_hour'],
                self.fuel_moistures.loc[i]['100_hour'],
                self.fuel_moistures.loc[i]['live_herbaceous'],
                self.fuel_moistures.loc[i]['live_woody']))


    def __writeInputSpotting(self, file):
        """Write spotting parameters to the input file."""
        file.write("FARSITE_SPOT_GRID_RESOLUTION: {0:.1f}\n".format(self.spot_grid_resolution))
        file.write("FARSITE_SPOT_PROBABILITY: {0:0.2f}\n".format(self.spot_probability))
        file.write("FARSITE_SPOT_IGNITION_DELAY: {0}\n".format(self.spot_ignition_delay))
        file.write("FARSITE_MINIMUM_SPOT_DISTANCE: {0}\n".format(self.minimum_spot_distance))
        file.write("FARSITE_ACCELERATION_ON: {0}\n".format(int(self.acceleration_on)))
        file.write("SPOTTING_SEED: {0}\n".format(self.spotting_seed))
    

    def __writeInputWeather(self, file):
        """Write weather parameters to the input file."""
        file.write("RAWS_FILE: {0}\n".format(self.name+".raws"))
    

    def __writeInputCrown(self, file):
        """Write crown fire parameters to the input file."""
        file.write("FOLIAR_MOISTURE_CONTENT: {0}\n".format(self.foliar_moisture_content))
        file.write("CROWN_FIRE_METHOD: " + self.crown_fire_method.name.title() + "\n")
    

    def writeInput(self, filename):
        """Write the FARSITE input file"""
        with open(filename, "w") as file:
            self.__writeInputHeader(file)
            file.write("\n\n")
            if self.burn_periods_count > 0:
                self.__writeInputBurnPeriods(file)
                file.write("\n\n")
            self.__writeInputSpotting(file)
            file.write("\n\n")
            self.__writeInputFuelMoistures(file)
            file.write("\n\n")
            self.__writeInputWeather(file)
            file.write("\n\n")
            self.__writeInputCrown(file)
    

    def __stripComment(self, line):
        """Remove comment from input file line."""
        return line.split("#")[0]
    

    def __detectLineType(self, line):
        """Detect the type of the given input file line."""
        line_nocomment = self.__stripComment(line)
        if line_nocomment in ("", "\n"):
            return LineType.EMPTY
        elif line_nocomment[0:19] == "FARSITE INPUTS FILE":
            return LineType.TITLE
        elif ":" in line_nocomment:
            return LineType.NAME_VALUE
        elif len(line_nocomment.split(" ")) >= 2:
            return LineType.LIST
        else:
            raise IOError("Unrecognized line format: " + line)
    

    def __parseNameValLine(self, line):
        """Parse input file line containing name-value pair."""
        line_nocomment = self.__stripComment(line)
        data = line_nocomment.strip().split(": ")
        if len(data) == 1:
            warnings.warn("No value for entry: " + data[0][:-1])
            [name, val] = [data[0], ""]
        else:
            [name, val] = data
        return [name, val]
    

    def __parseListLine(self, line):
        """Parse input file line containing list."""
        line_nocomment = self.__stripComment(line)
        return line_nocomment.strip().split(" ")
    

    def __parseTime(self, timestr):
        """Parse datetime from string."""
        vals = timestr.split(" ")
        return dt.datetime(
            _DEFAULT_YEAR,
            int(vals[0]),
            int(vals[1]),
            int(vals[2][0:2]),
            int(vals[2][2:4]))
    

    def __readNameValLine(self, line):
        """Read the name-value pair line from the input file and set the appropriate attribute."""
        [name, val] = self.__parseNameValLine(line)
        if name == "FARSITE_START_TIME":
            self.start_time = self.__parseTime(val)
        elif name == "FARSITE_END_TIME":
            self.end_time = self.__parseTime(val)
        elif name == "FARSITE_TIMESTEP":
            self.timestep = int(val)
        elif name == "FARSITE_DISTANCE_RES":
            self.distance_res = float(val)
        elif name == "FARSITE_PERIMETER_RES":
            self.perimeter_res = float(val)
        elif name == "FARSITE_MIN_IGNITION_VERTEX_DISTANCE":
            self.min_ignition_vertex_distance = float(val)
        elif name == "FARSITE_SPOT_GRID_RESOLUTION":
            self.spot_grid_resolution = float(val)
        elif name == "FARSITE_SPOT_PROBABILITY":
            self.spot_probability = float(val)
        elif name == "FARSITE_SPOT_IGNITION_DELAY":
            self.spot_ignition_delay = int(val)
        elif name == "FARSITE_MINIMUM_SPOT_DISTANCE":
            self.minimum_spot_distance = int(val)
        elif name == "FARSITE_ACCELERATION_ON":
            self.acceleration_on = bool(int(val))
        elif name == "SPOTTING_SEED":
            self.spotting_seed = int(val)
        elif name == "RAWS_FILE":
            self.weather = raws.RAWS(os.path.join(self.root_dir, val))
            self.start_time = self.start_time.replace(year=self.weather.data.at[0, 'time'].year)
            self.end_time = self.end_time.replace(year=self.weather.data.at[0, 'time'].year)
        elif name == "FOLIAR_MOISTURE_CONTENT":
            self.foliar_moisture_content = int(val)
        elif name == "CROWN_FIRE_METHOD":
            self.crown_fire_method = CrownFireMethod[val.upper()]
        elif name == "NUMBER_PROCESSORS":
            self.number_processors = int(val)
    

    def __readBurnPeriodsLines(self, lines):
        """Read the table of burn periods from the input file."""
        for line in lines:
            vals = self.__parseListLine(line)
            entry = {}
            entry['start'] = dt.datetime(
                year   = _DEFAULT_YEAR,
                month  = int(vals[0]),
                day    = int(vals[1]),
                hour   = int(vals[2][0:2]),
                minute = int(vals[2][2:4]))
            entry['end'] = dt.datetime(
                year   = _DEFAULT_YEAR,
                month  = int(vals[0]),
                day    = int(vals[1]),
                hour   = int(vals[3][0:2]),
                minute = int(vals[3][2:4]))
            self.burn_periods = self.burn_periods.append(entry, ignore_index=True)
    

    def __readMoistureLines(self, lines):
        """Read the table of fuel moistures data from the input file."""
        for line in lines:
            vals = self.__parseListLine(line)
            entry = {}
            entry['model']           = int(vals[0])
            entry['1_hour']          = int(vals[1])
            entry['10_hour']         = int(vals[2])
            entry['100_hour']        = int(vals[3])
            entry['live_herbaceous'] = int(vals[4])
            entry['live_woody']      = int(vals[5])
            self.fuel_moistures = self.fuel_moistures.append(entry, ignore_index=True)
    

    def readInput(self, filename):
        """Read the FARSITE input file"""
        self.fuel_moistures = self.fuel_moistures[0:0]
        with open(filename, "r") as file:
            for line in file:
                line_type = self.__detectLineType(line)
                if line_type in (LineType.TITLE, LineType.EMPTY):
                    continue
                elif line_type == LineType.NAME_VALUE:
                    [name, val] = self.__parseNameValLine(line)
                    if name == "FARSITE_BURN_PERIODS":
                        self.burn_periods_count = int(val)
                        lines = []
                        for i in range(self.burn_periods_count):
                            lines.append(file.readline())
                        self.__readBurnPeriodsLines(lines)
                    if name == "FUEL_MOISTURES_DATA":
                        self.fuel_moistures_count = int(val)
                        lines = []
                        for i in range(self.fuel_moistures_count):
                            lines.append(file.readline())
                        self.__readMoistureLines(lines)
                    else:
                        self.__readNameValLine(line)
    

    def write(self):
        """Write all files to the root directory with proper structure"""
        landscape_dir = os.path.join(self.root_dir, self.landscape_dir_local)
        ignition_dir = os.path.join(self.root_dir, self.ignition_dir_local)
        out_dir = os.path.join(self.root_dir, self.out_dir_local)

        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        if not os.path.isdir(landscape_dir):
            os.makedirs(landscape_dir, exist_ok=True)
        if not os.path.isdir(ignition_dir):
            os.makedirs(ignition_dir, exist_ok=True)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        input_file_local = self.name+".input"
        lcp_prefix_local = os.path.join(self.landscape_dir_local, self.name)
        weather_file_local = self.name+".raws"
        ignition_file_local = os.path.join(self.ignition_dir_local, self.name+".shp")
        run_file_local = "run_"+self.name+".txt"
        job_file_local = "job.slurm"
        out_prefix_local = os.path.join(self.out_dir_local, self.name)

        input_file = os.path.join(self.root_dir, input_file_local)
        lcp_prefix = os.path.join(self.root_dir, lcp_prefix_local)
        weather_file = os.path.join(self.root_dir, weather_file_local)
        ignition_file = os.path.join(self.root_dir, ignition_file_local)
        run_file = os.path.join(self.root_dir, run_file_local)
        job_file = os.path.join(self.root_dir, job_file_local)
        
        self.writeInput(input_file)
        self.lcp.write(lcp_prefix)
        self.weather.write(weather_file)
        self.ignition.to_file(ignition_file)
        self.sbatch.runfile_name_local = run_file_local
        self.sbatch.write(job_file)

        with open(run_file, "w") as file:
            file.write("{0} {1} {2} {3} {4} {5}".format(
                lcp_prefix_local+".lcp",
                input_file_local,
                ignition_file_local,
                0,
                out_prefix_local,
                self.out_type))
    

    def run(self):
        """Submit the slurm job to run the case."""
        current_dir = os.getcwd()
        os.chdir(self.root_dir)
        result = subprocess.run(["sbatch", self.jobfile_name_local], stdout=subprocess.PIPE)
        self.job_id = int(result.stdout.decode('utf-8').strip().split(" ")[-1])
        os.chdir(current_dir)
    

    def logFile(self):
        """Find log file, if it exists."""
        files = os.listdir(self.root_dir)
        log_file = None
        for file in files:
            if "{0:d}.out".format(self.job_id) in file:
                return file
    

    def isStarted(self):
        """Determine whether the FARSITE simulation has started."""
        # If job ID has not been set, run has not started
        if not self.job_id:
            return False

        # If log file does not exist, run has not started
        log_file = self.logFile()
        if not log_file:
            return False
        
        # Read log file for launch line, started if found
        with open(os.path.join(self.root_dir, log_file), "r") as file:
            for line in file:
                if "Launching Farsite" in line:
                    return True
        
        # Not started if line not found
        return False
    

    def isDone(self):
        """Determine whether the slurm job has completed."""
        # FARSITE cannot be done if it has not started
        if not self.isStarted():
            return False
        
        # Read log file for output line, done if found
        with open(os.path.join(self.root_dir, self.logFile()), "r") as file:
            for line in file:
                if "Writing outputs" in line:
                    return True
        
        # Not done if output line not found
        return False
    

    def ignitionFailed(self):
        """Determine whether the case has failed to ignite."""
        # Ignition cannot have failed yet if FARSITE has not started
        if not self.isStarted():
            return False
        
        # Read log file for ignition failure line, failed if found
        with open(os.path.join(self.root_dir, self.logFile()), "r") as file:
            for line in file:
                if "No ignition" in line:
                    return True
        
        # Not failed if failure line not found
        return False
    

    def __outputFile(self, name):
        """Build output file path string."""
        return os.path.join(
            self.root_dir,
            self.out_dir_local,
            self.name + "_" + name)
    

    def __convertAndMergePerimeters(self):
        """Convert LineString perimeters to Polygon and MultiPolygon objects."""
        self.perimeters_merged = self.perimeters.copy()
        self.perimeters_merged = self.perimeters_merged[0:0]

        elapsed_minutes = self.perimeters['Elapsed_Mi'][0]
        polys = []

        for i in range(len(self.perimeters)):
            # If time has changed, merge existing polygons, write to merged object, and empty list
            if not np.isclose(self.perimeters['Elapsed_Mi'][i], elapsed_minutes):
                self.perimeters_merged = self.perimeters_merged.append(
                    self.perimeters.loc[i-1],
                    ignore_index=True)
                self.perimeters_merged.iat[-1, -1] = unary_union(polys)
                polys = []
                elapsed_minutes = self.perimeters['Elapsed_Mi'][i]
            
            # Add current polygon to list
            poly = geometry.Polygon(self.perimeters.geometry[i])
            if not poly.is_valid:
                poly = poly.buffer(0)
            polys.append(poly)
        
        # Merge remaining polygons and write to merged object
        self.perimeters_merged = self.perimeters_merged.append(
            self.perimeters.loc[i],
            ignore_index=True)
        self.perimeters_merged.iat[-1, -1] = unary_union(polys)


    def readOutput(self):
        """Read FARSITE output data."""
        self.arrival_time       = ascii_data.ASCIIData(self.__outputFile("ArrivalTime.asc"))
        self.crown_fire         = ascii_data.ASCIIData(self.__outputFile("CrownFire.asc"))
        self.flame_length       = ascii_data.ASCIIData(self.__outputFile("FlameLength.asc"))
        self.heat_per_unit_area = ascii_data.ASCIIData(self.__outputFile("HeatPerUnitArea.asc"))
        self.ignitions          = ascii_data.ASCIIData(self.__outputFile("Ignitions.asc"))
        self.intensity          = ascii_data.ASCIIData(self.__outputFile("Intensity.asc"))
        self.reaction_intensity = ascii_data.ASCIIData(self.__outputFile("ReactionIntensity.asc"))
        self.spot_grid          = ascii_data.ASCIIData(self.__outputFile("SpotGrid.asc"))
        self.spread_direction   = ascii_data.ASCIIData(self.__outputFile("SpreadDirection.asc"))
        self.spread_rate        = ascii_data.ASCIIData(self.__outputFile("SpreadRate.asc"))
        self.spots = gpd.read_file(self.__outputFile("Spots.shp"))
        self.perimeters = gpd.read_file(self.__outputFile("Perimeters.shp"))
        self.__convertAndMergePerimeters()
    

    def __plotPerimeters(self, ax, n_perimeters):
        """Add perimeters to plot."""
        plot_interval = int(np.round(len(self.perimeters_merged.geometry) / n_perimeters))
        for i, per in enumerate(self.perimeters_merged.geometry):
            if i % plot_interval:
                continue

            if isinstance(per, geometry.MultiPolygon):
                polys = list(per)
            else:
                polys = [per]
            
            # import code; code.interact(local=locals())

            for poly in polys:
                x, y = poly.exterior.xy
                x_scaled = np.array(x) / 1000.0
                y_scaled = np.array(y) / 1000.0
                ax.plot(x_scaled, y_scaled, color='k', linewidth=1)
    

    def renderOutput(self, filename):
        """Plot fire and save figure to file."""
        fig, axs = plt.subplots(2, 2)
        extent = (
            self.lcp.utm_west  / 1000.0,
            self.lcp.utm_east  / 1000.0,
            self.lcp.utm_south / 1000.0,
            self.lcp.utm_north / 1000.0)
        
        im1 = axs[0,0].imshow(self.arrival_time.data, extent=extent)
        plt.colorbar(im1, ax=axs[0,0], label="Arrival Time (min)")

        im2 = axs[0,1].imshow(self.lcp.layers['elevation'].data, extent=extent)
        plt.colorbar(im2, ax=axs[0,1], label="Elevation (m)")

        im3 = axs[1,0].imshow(self.lcp.layers['fuel'].data, extent=extent)
        plt.colorbar(im3, ax=axs[1,0], label="Fuel Model")

        im4 = axs[1,1].imshow(self.spread_rate.data, extent=extent)
        plt.colorbar(im4, ax=axs[1,1], label="Spread Rate (km/h)")

        for ax in axs.reshape(-1):
            self.__plotPerimeters(ax, 5)
            ax.set_xlabel("y (km)")
            ax.set_ylabel("x (km)")
        
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    

    def __burnMap(self, perimeter_poly):
        """Compute the fractional burn map for a given perimeter."""
        burn = np.zeros([self.lcp.num_north, self.lcp.num_east])
        prepared_perimeter = prep(perimeter_poly)

        # Iterate over cells
        for i in range(self.lcp.num_north):
            for j in range(self.lcp.num_east):
                cell = geometry.box(
                    (j)   * self.lcp.res_x + self.lcp.utm_west,
                    (i)   * self.lcp.res_y + self.lcp.utm_south,
                    (j+1) * self.lcp.res_x + self.lcp.utm_west,
                    (i+1) * self.lcp.res_y + self.lcp.utm_south)
                if prepared_perimeter.contains(cell):
                    burn[i,j] = 1.0
                elif prepared_perimeter.intersects(cell):
                    burn[i,j] = cell.intersection(perimeter_poly).area / cell.area

        return burn
    

    def computeBurnMaps(self):
        """Compute the fractional burn maps for all output perimeters."""
        n_steps = len(self.perimeters_merged)
        self.burn = np.zeros([
            n_steps,
            self.lcp.num_north,
            self.lcp.num_east])
        for i in range(n_steps):
            if self.verbose:
                print("Computing case " + self.name + " burn map {0}/{1}".format(i+1, n_steps))
            self.burn[i] = self.__burnMap(self.perimeters_merged.loc[i].geometry)


    def getOutputTimes(self):
        """Get output times from perimeters table."""
        times = []
        for i, entry in self.perimeters_merged.iterrows():
            times.append(dt.datetime(
                self.start_time.year,
                int(entry['Month']),
                int(entry['Day']),
                int("{0:04d}".format(int(entry['Hour']))[0:2]),
                int("{0:04d}".format(int(entry['Hour']))[2:4])))
        return times
    

    def writeMoistureNPY(self, prefix):
        """Write moisture data in Numpy format."""
        nonzero = np.where(self.lcp.layers['fuel'].vals)[0]
        if nonzero.size == 0:
            models = [0]
        else:
            n_models = nonzero[-1] + 1
            models = self.lcp.layers['fuel'].vals[0:n_models]
        
        # TODO - convert to dict

        moisture_1_hour          = np.zeros([self.lcp.num_north, self.lcp.num_east], dtype=np.int16)
        moisture_10_hour         = np.zeros([self.lcp.num_north, self.lcp.num_east], dtype=np.int16)
        moisture_100_hour        = np.zeros([self.lcp.num_north, self.lcp.num_east], dtype=np.int16)
        moisture_live_herbaceous = np.zeros([self.lcp.num_north, self.lcp.num_east], dtype=np.int16)
        moisture_live_woody      = np.zeros([self.lcp.num_north, self.lcp.num_east], dtype=np.int16)

        # Iterate over models in landscape
        for model in models:
            # Find where model exists in landscape
            mask_layer = self.lcp.layers['fuel'].data == model

            # Find moisture data entry for model
            if model in list(self.fuel_moistures['model']):
                mask_moisture = self.fuel_moistures['model'] == model
            else:
                mask_moisture = self.fuel_moistures['model'] == 0 # Default to data for model 0

            # Set moistures in given locations to given model data
            moisture_1_hour         [mask_layer] = int(self.fuel_moistures['1_hour'         ].loc[mask_moisture])
            moisture_10_hour        [mask_layer] = int(self.fuel_moistures['10_hour'        ].loc[mask_moisture])
            moisture_100_hour       [mask_layer] = int(self.fuel_moistures['100_hour'       ].loc[mask_moisture])
            moisture_live_herbaceous[mask_layer] = int(self.fuel_moistures['live_herbaceous'].loc[mask_moisture])
            moisture_live_woody     [mask_layer] = int(self.fuel_moistures['live_woody'     ].loc[mask_moisture])
        
        # Write
        np.save(prefix + "_moisture_1_hour.npy",          moisture_1_hour         )
        np.save(prefix + "_moisture_10_hour.npy",         moisture_10_hour        )
        np.save(prefix + "_moisture_100_hour.npy",        moisture_100_hour       )
        np.save(prefix + "_moisture_live_herbaceous.npy", moisture_live_herbaceous)
        np.save(prefix + "_moisture_live_woody.npy",      moisture_live_woody     )


    def exportData(self, prefix):
        """Export data in Numpy format."""
        [out_dir, out_name] = os.path.split(prefix)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self.lcp.writeNPY(prefix)
        self.weather.writeWindNPY(
            prefix,
            (self.lcp.num_north, self.lcp.num_east),
            self.getOutputTimes())
        self.writeMoistureNPY(prefix)

        if self.burn is not None:
            np.save(prefix + "_burn.npy", self.burn)
        else:
            raise RuntimeError("Burn maps have not yet been computed -- run Case.computeBurnMaps()")


def main():
    pass


if __name__ == "__main__":
    main()
