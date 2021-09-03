"""Class representing a single Farsite simulation."""
import os
import warnings
from enum import Enum
import datetime as dt
import pandas as pd

import landscape
import raws
import ignition
import sbatch
import ascii_data
import generate as gen


class CrownFireMethod(Enum):
    FINNEY = 1
    REINHARDT = 2


class LineType(Enum):
    TITLE = 1
    NAME_VALUE = 2
    EMPTY = 3


class Case:
    def __init__(self, jobfile_name=None):
        # Inputs
        self.name = "case"
        self.root_dir = "./"
        self.out_dir_local = "output/"
        self.landscape_dir_local = "landscape/"
        self.ignition_dir_local = "ignition/"
        self.jobfile_name_local = "job.slurm"
        self.start_time = dt.datetime(2000, 1, 1)
        self.end_time = dt.datetime(2000, 1, 1)
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
        self.fuel_moistures_data = 0
        self.fuel_moistures = pd.DataFrame(columns = [
            "model",
            "1_hour",
            "10_hour",
            "100_hour",
            "live_herbaceous",
            "live_woody"
        ])
        self.weather = raws.RAWS()
        self.foliar_moisture_content = 100
        self.crown_fire_method = CrownFireMethod.FINNEY
        self.number_processors = 1
        self.lcp = landscape.Landscape()
        self.ignit = ignition.Ignition()
        self.out_type = 0
        self.sbatch = sbatch.SBatch()

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

        if jobfile_name:
            self.read(jobfile_name)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def setName(self, name):
        self.name = name
        self.sbatch.set_option("-J", name)
    

    def read(self, jobfile_name):
        [self.root_dir, self.jobfile_name_local] = os.path.split(jobfile_name)
        self.sbatch.read(jobfile_name)
        runfile_name = os.path.join(self.root_dir, self.sbatch.runfile_name_local)
        with open(runfile_name, "r") as file:
            line = file.readline()
        args = line.split(" ")
        self.lcp.read(os.path.join(self.root_dir, os.path.splitext(args[0])[0]))
        self.readInput(os.path.join(self.root_dir, args[1]))
        self.ignit.read(os.path.join(self.root_dir, args[2]))
        out_prefix_local = args[4]
        [self.out_dir_local, name] = os.path.split(out_prefix_local)
        self.setName(name)
        self.out_type = int(args[5])
    

    def __writeInputHeader(self, file):
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
        file.write("FARSITE_TIMESTEP: {0}\n".format(self.timestep))
        file.write("FARSITE_DISTANCE_RES: {0:.1f}\n".format(self.distance_res))
        file.write("FARSITE_PERIMETER_RES: {0:.1f}\n".format(self.perimeter_res))
        file.write("FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {0:.1f}\n".format(self.min_ignition_vertex_distance))
        file.write("NUMBER_PROCESSORS: {0}\n".format(self.number_processors))
    

    def __writeInputMoisture(self, file):
        file.write("FUEL_MOISTURES_DATA: {0}\n".format(self.fuel_moistures_data))
        for i in range(len(self.fuel_moistures)):
            file.write("{0} {1} {2} {3} {4} {5}\n".format(
                self.fuel_moistures.loc[i]['model'],
                self.fuel_moistures.loc[i]['1_hour'],
                self.fuel_moistures.loc[i]['10_hour'],
                self.fuel_moistures.loc[i]['100_hour'],
                self.fuel_moistures.loc[i]['live_herbaceous'],
                self.fuel_moistures.loc[i]['live_woody'],
            ))
    

    def __writeInputSpotting(self, file):
        file.write("FARSITE_SPOT_GRID_RESOLUTION: {0:.1f}\n".format(self.spot_grid_resolution))
        file.write("FARSITE_SPOT_PROBABILITY: {0:0.2f}\n".format(self.spot_probability))
        file.write("FARSITE_SPOT_IGNITION_DELAY: {0}\n".format(self.spot_ignition_delay))
        file.write("FARSITE_MINIMUM_SPOT_DISTANCE: {0}\n".format(self.minimum_spot_distance))
        file.write("FARSITE_ACCELERATION_ON: {0}\n".format(int(self.acceleration_on)))
        file.write("SPOTTING_SEED: {0}\n".format(self.spotting_seed))
    

    def __writeInputWeather(self, file):
        file.write("RAWS_FILE: {0}\n".format(self.name+".raws"))
    

    def __writeInputCrown(self, file):
        file.write("FOLIAR_MOISTURE_CONTENT: {0}\n".format(self.foliar_moisture_content))
        file.write("CROWN_FIRE_METHOD: " + self.crown_fire_method.name.title() + "\n")
    

    def writeInput(self, filename):
        """Write Farsite input file"""
        with open(filename, "w") as file:
            self.__writeInputHeader(file)
            file.write("\n\n")
            self.__writeInputSpotting(file)
            file.write("\n\n")
            self.__writeInputMoisture(file)
            file.write("\n\n")
            self.__writeInputWeather(file)
            file.write("\n\n")
            self.__writeInputCrown(file)
    

    def __stripComment(self, line):
        return line.split("#")[0]
    

    def __detectLineType(self, line):
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
        line_nocomment = self.__stripComment(line)
        data = line_nocomment.strip().split(": ")
        if len(data) == 1:
            warnings.warn("No value for entry: " + data[0][:-1])
            [name, val] = [data[0], ""]
        else:
            [name, val] = data
        return [name, val]
    

    def __parseListLine(self, line):
        line_nocomment = self.__stripComment(line)
        return line_nocomment.strip().split(" ")
    

    def __parseTime(self, timestr):
        vals = timestr.split(" ")
        return dt.datetime(
            2000,
            int(vals[0]),
            int(vals[1]),
            int(vals[2][0:2]),
            int(vals[2][2:4]))
    

    def __readNameValLine(self, line):
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
        elif name == "FOLIAR_MOISTURE_CONTENT":
            self.foliar_moisture_content = int(val)
        elif name == "CROWN_FIRE_METHOD":
            self.crown_fire_method = CrownFireMethod[val.upper()]
        elif name == "NUMBER_PROCESSORS":
            self.number_processors = int(val)
    

    def __readMoistureLines(self, lines):
        for line in lines:
            vals = self.__parseListLine(line)
            entry = {}
            entry['model']           = vals[0]
            entry['1_hour']          = vals[1]
            entry['10_hour']         = vals[2]
            entry['100_hour']        = vals[3]
            entry['live_herbaceous'] = vals[4]
            entry['live_woody']      = vals[5]
            self.fuel_moistures = self.fuel_moistures.append(entry, ignore_index=True)
    

    def readInput(self, filename):
        """Read Farsite input file"""
        self.fuel_moistures = self.fuel_moistures[0:0]
        with open(filename, "r") as file:
            for line in file:
                line_type = self.__detectLineType(line)
                if line_type in (LineType.TITLE, LineType.EMPTY):
                    continue
                elif line_type == LineType.NAME_VALUE:
                    [name, val] = self.__parseNameValLine(line)
                    if name == "FUEL_MOISTURES_DATA":
                        self.fuel_moistures_data = int(val)
                        lines = []
                        for i in range(self.fuel_moistures_data):
                            lines.append(file.readline())
                        self.__readMoistureLines(lines)
                    else:
                        self.__readNameValLine(line)
    

    def write(self):
        """Write all files to directory with proper structure"""

        landscape_dir = os.path.join(self.root_dir, self.landscape_dir_local)
        ignition_dir = os.path.join(self.root_dir, self.ignition_dir_local)
        out_dir = os.path.join(self.root_dir, self.out_dir_local)

        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.isdir(landscape_dir):
            os.mkdir(landscape_dir)
        if not os.path.isdir(ignition_dir):
            os.mkdir(ignition_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        input_file_local = self.name+".input"
        lcp_prefix_local = os.path.join(self.landscape_dir_local, self.name)
        weather_file_local = self.name+".raws"
        ignit_file_local = os.path.join(self.ignition_dir_local, self.name+".shp")
        run_file_local = "run_"+self.name+".txt"
        job_file_local = "job.slurm"
        out_prefix_local = os.path.join(self.out_dir_local, self.name)

        input_file = os.path.join(self.root_dir, input_file_local)
        lcp_prefix = os.path.join(self.root_dir, lcp_prefix_local)
        weather_file = os.path.join(self.root_dir, weather_file_local)
        ignit_file = os.path.join(self.root_dir, ignit_file_local)
        run_file = os.path.join(self.root_dir, run_file_local)
        job_file = os.path.join(self.root_dir, job_file_local)
        
        self.writeInput(input_file)
        self.lcp.write(lcp_prefix)
        self.weather.write(weather_file)
        self.ignit.write(ignit_file)
        self.sbatch.runfile_name_local = run_file_local
        self.sbatch.write(job_file)

        with open(run_file, "w") as file:
            file.write("{0} {1} {2} {3} {4} {5}".format(
                lcp_prefix_local+".lcp",
                input_file_local,
                ignit_file_local,
                0,
                out_prefix_local,
                self.out_type))
    

    def run(self):
        os.chdir(self.root_dir)
        os.system("sbatch " + self.jobfile_name_local)
    

    def __outputFile(self, name):
        return os.path.join(
            self.root_dir,
            self.out_dir_local,
            self.name + "_" + name + ".asc")
    

    def readOutput(self):
        self.arrival_time       = ascii_data.ASCIIData(self.__outputFile("ArrivalTime"))
        self.crown_fire         = ascii_data.ASCIIData(self.__outputFile("CrownFire"))
        self.flame_length       = ascii_data.ASCIIData(self.__outputFile("FlameLength"))
        self.heat_per_unit_area = ascii_data.ASCIIData(self.__outputFile("HeatPerUnitArea"))
        self.ignitions          = ascii_data.ASCIIData(self.__outputFile("Ignitions"))
        self.intensity          = ascii_data.ASCIIData(self.__outputFile("Intensity"))
        self.reaction_intensity = ascii_data.ASCIIData(self.__outputFile("ReactionIntensity"))
        self.spot_grid          = ascii_data.ASCIIData(self.__outputFile("SpotGrid"))
        self.spread_direction   = ascii_data.ASCIIData(self.__outputFile("SpreadDirection"))
        self.spread_rate        = ascii_data.ASCIIData(self.__outputFile("SpreadRate"))


def main():
    pass


if __name__ == "__main__":
    main()
