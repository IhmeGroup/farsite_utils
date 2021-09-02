import os
import warnings
from enum import Enum
import datetime as dt
import pandas as pd

import landscape as ls
import raws
import ignition
import generate as gen

class CrownFireMethod(Enum):
    FINNEY = 1
    REINHARDT = 2


class LineType(Enum):
    TITLE = 1
    NAME_VALUE = 2
    EMPTY = 3


class Case:
    def __init__(self, runfile=None):
        self.root_dir = "./"
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
        self.lcp = ls.Landscape()
        self.ignit = ignition.Ignition()
        self.out_prefix = ""
        self.out_type = 0

        if runfile:
            self.read(runfile)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def read(self, runfilename):
        [self.root_dir, name] = os.path.split(runfilename)
        with open(runfilename, "r") as file:
            line = file.readline()
        args = line.split(" ")
        self.lcp = ls.Landscape(os.path.join(self.root_dir, os.path.splitext(args[0])[0]))
        self.readInput(os.path.join(self.root_dir, args[1]))
        self.ignit = ignition.Ignition(os.path.join(self.root_dir, args[2]))
        self.out_prefix = args[4]
        self.out_type = int(args[5])
    

    def __writeInputHeader(self, file):
        file.write("FARSITE INPUTS FILE VERSION 1.0\n")
        file.write("FARSITE_START_TIME: {0} {1} {2}{3:02d}\n".format(
            self.start_time.month,
            self.start_time.day,
            self.start_time.hour,
            self.start_time.minute))
        file.write("FARSITE_END_TIME: {0} {1} {2}{3:02d}\n".format(
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
    

    def write(self, prefix):
        """Write all files to directory with proper structure"""

        [root_dir, name] = os.path.split(prefix)
        [out_dir_local, name] = os.path.split(self.out_prefix)

        landscape_dir = os.path.join(root_dir, "landscape/")
        ignition_dir = os.path.join(root_dir, "ignition/")
        out_dir = os.path.join(root_dir, out_dir_local)

        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        if not os.path.isdir(landscape_dir):
            os.mkdir(landscape_dir)
        if not os.path.isdir(ignition_dir):
            os.mkdir(ignition_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        input_file = prefix+".input"
        lcp_file = os.path.join(landscape_dir, name)
        weather_file = prefix+".raws"
        ignit_file = os.path.join(ignition_dir, name+".shp")
        run_file = os.path.join(root_dir, "run_"+name+".txt")
        
        self.writeInput(input_file)
        self.lcp.write(lcp_file)
        self.weather.write(weather_file)
        self.ignit.write(ignit_file)

        with open(run_file, "w") as file:
            file.write("{0} {1} {2} {3} {4} {5}".format(
                lcp_file,
                input_file,
                ignit_file,
                0,
                self.out_prefix,
                self.out_type))


def main():
    pass


if __name__ == "__main__":
    main()
