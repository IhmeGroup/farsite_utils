import os
from enum import Enum
import pandas as pd

import landscape as ls
import raws
import ignition
import generate as gen

class CrownFireMethod(Enum):
    FINNEY = 1
    REINHARDT = 2

class FarsiteCase:
    def __init__(self, prefix=None):
        self.prefix = prefix
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
        self.fuel_moistures_data = True
        self.fuel_moistures = pd.DataFrame(columns = [
            "model",
            "1_hour",
            "10_hour",
            "100_hour",
            "live_herbacious",
            "live_woody"
        ])
        self.weather = raws.RAWS()
        self.foliar_moisture_content = 100
        self.crown_fire_method = CrownFireMethod.FINNEY
        self.number_processors = 1
        self.lcp = ls.Landscape(prefix)
        self.ignit = ignition.Ignition()

        if self.prefix:
            self.read(self.prefix)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def read(self, prefix):
        """Read case from files"""
        self.weather = raws.RAWS(prefix + ".raws")
        self.lcp = ls.Landscape(prefix)
        self.ignit = ignition.Ignition(prefix + ".shp")
        raise NotImplementedError
    

    def writeRun(self, filename):
        """Write Farsite run file"""
        raise NotImplementedError
    

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
        file.write("FARSITE_DISTANCE_RES: {0}\n".format(self.distance_res))
        file.write("FARSITE_PERIMETER_RES: {0}\n".format(self.perimeter_res))
        file.write("FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {0}\n".format(self.min_ignition_vertex_distance))
        file.write("NUMBER_PROCESSORS: {0}\n".format(self.number_processors))
    

    def __writeInputMoisture(self, file):
        file.write("FUEL_MOISTURES_DATA: {0}\n".format(int(self.fuel_moistures_data)))
        for i in range(len(self.fuel_moistures)):
            file.write("{0} {1} {2} {3} {4} {5}\n".format(
                self.fuel_moistures[i]['model'],
                self.fuel_moistures[i]['1_hour'],
                self.fuel_moistures[i]['10_hour'],
                self.fuel_moistures[i]['100_hour'],
                self.fuel_moistures[i]['live_herbaceous'],
                self.fuel_moistures[i]['live_woody'],
            ))
    

    def __writeInputSpotting(self, file):
        file.write("FARSITE_SPOT_GRID_RESOLUTION: {0}\n".format(self.spot_grid_resolution))
        file.write("FARSITE_SPOT_PROBABILITY: {0:0.2f}\n".format(self.spot_probability))
        file.write("FARSITE_SPOT_IGNITION_DELAY: {0}\n".format(self.spot_ignition_delay))
        file.write("FARSITE_MINIMUM_SPOT_DISTANCE: {0}\n".format(self.minimum_spot_distance))
        file.write("FARSITE_ACCELERATION_ON: {0}\n".format(self.acceleration_on))
        file.write("SPOTTING_SEED: {0}\n".format(self.spotting_seed))
    

    def __writeInputCrown(self, file):
        file.write("FOLIAR_MOISTURE_CONTENT: {0}\n".format(self.foliar_moisture_content))
        file.write("CROWN_FIRE_METHOD: " + self.crown_fire_method.name.title() + "\n")
    

    def writeInput(self, filename):
        """Write Farsite input file"""
        with open(filename, "w") as file:
            self.__writeInputHeader(file)
            self.__writeInputSpotting(file)
            file.write("\n\n")
            file.write("RAWS_FILE: " + self.raws.filename)
            file.write("\n\n")
            self.__writeInputCrown(file)
    

    def writeRun(self, filename):
        """Write Farsite run file"""
        raise NotImplementedError
    

    def writeDir(self, prefix):
        """Write all files to directory with proper structure"""

        [directory, name] = os.path.split(prefix)

        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        self.writeInput(prefix + ".input")
        self.lcp.writeLCP(os.path.join(directory, "landscape/", name+".lcp"))
        self.weather.write(prefix + ".raws")
        self.ignit.write(os.path.join(directory, "ignition/", name+".shp"))
        self.writeRun(os.path.join(directory, "run_"+name+".txt")

        raise NotImplementedError