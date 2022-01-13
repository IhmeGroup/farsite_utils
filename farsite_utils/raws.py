"""Utilities for reading and writing RAWS files."""

from enum import Enum
import datetime as dt
import numpy as np
import pandas as pd
from collections import Counter


_HEADER_LINES = 3
_DATA_COLS = [
    'time',
    'temperature',
    'humidity',
    'precipitation',
    'wind_speed',
    'wind_direction',
    'cloud_cover']


class Unit(Enum):
    ENGLISH = 1
    METRIC = 2


class RAWS:
    def __init__(self, filename=None):
        self.elevation = 0
        self.units = Unit.ENGLISH
        self.count = 0
        self.data = pd.DataFrame(columns = _DATA_COLS)

        if filename:
            self.read(filename)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    
    
    @property
    def data(self):
        return self._data
    

    @data.setter
    def data(self, value):
        if not (Counter(value.columns) == Counter(_DATA_COLS)):
            raise ValueError(
                "Data must have the following columns: " +
                [i + ", " for i in _DATA_COLS[:-1]] +
                _DATA_COLS[-1])
        self._data = value
        self.count = len(value.index)
    

    def __parseHeaderLine(self, line):
        return line.strip().split(": ")
    

    def __parseHeader(self, file):
        # Parse elevation
        line1 = file.readline()
        [var1, val1] = self.__parseHeaderLine(line1)
        if var1 == "RAWS_ELEVATION": 
            self.elevation = int(val1)
        else:
            raise IOError("File is improperly formatted")

        # Parse unit system
        line2 = file.readline()
        [var2, val2] = self.__parseHeaderLine(line2)
        if var2 == "RAWS_UNITS":
            try:
                self.units = Unit[val2.upper()]
            except KeyError:
                raise IOError("Invalid units specification. Must be ENGLISH or METRIC.")
        else:
            raise IOError("File is improperly formatted")
        
        # Parse data entry count
        line3 = file.readline()
        [var3, val3] = self.__parseHeaderLine(line3)
        if var3 == "RAWS":
            self.count = int(val3)
        else:
            raise IOError("File is improperly formatted")
    

    def __parseBodyLine(self, line):
        vals = line.strip().split(" ")
        entry = {}
        entry['time'] = dt.datetime(
            int(vals[0]),
            int(vals[1]),
            int(vals[2]),
            int("{0:04d}".format(int(vals[3]))[0:2]),
            int("{0:04d}".format(int(vals[3]))[2:4]))
        entry['temperature'] =    int(vals[4])
        entry['humidity'] =       int(vals[5])
        entry['precipitation'] =  float(vals[6])
        entry['wind_speed'] =     int(vals[7])
        entry['wind_direction'] = int(vals[8])
        entry['cloud_cover'] =    int(vals[9])
        return entry
    

    def __parseBody(self, file):
        for i in range(self.count):
            line = file.readline()
            entry = self.__parseBodyLine(line)
            self._data = self._data.append(entry, ignore_index=True)


    def read(self, filename):
        self._data = self._data[0:0]
        with open(filename, "r") as file:
            self.__parseHeader(file)
            self.__parseBody(file)
    

    def __writeHeader(self, file):
        file.write("RAWS_ELEVATION: {0}\n".format(self.elevation))
        file.write("RAWS_UNITS: " + self.units.name.title() + "\n")
        file.write("RAWS: {0}\n".format(self.count))
    

    def __writeBodyLine(self, file, entry):
        file.write("{0} {1} {2} {3}{4:02d} {5} {6} {7:1.2f} {8} {9} {10}\n".format(
            entry['time'].year,
            entry['time'].month,
            entry['time'].day,
            entry['time'].hour,
            entry['time'].minute,
            entry['temperature'],
            entry['humidity'],
            entry['precipitation'],
            entry['wind_speed'],
            entry['wind_direction'],
            entry['cloud_cover']))
    

    def __writeBody(self, file):
        for i in range(self.count):
            self.__writeBodyLine(file, self._data.loc[i])
    
    
    def write(self, filename):
        with open(filename, "w") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
    

    def writeWindNPY(self, prefix, shape, query_times=None):
        if not query_times:
            query_times = [self._data.loc[0, 'time']]
        
        # Convert datetimes to elapsed seconds
        epoch = self._data.loc[0, 'time']
        query_times_secs = [(time - epoch).total_seconds() for time in query_times]
        raws_times_secs = [(time.to_pydatetime() - epoch).total_seconds() for time in self._data['time']]

        # Interpolate data at query times
        wind_speed     = np.interp(query_times_secs, raws_times_secs, list(self._data['wind_speed']))
        wind_direction = np.interp(query_times_secs, raws_times_secs, list(self._data['wind_direction']))

        # Compute components from speed and direction
        wind_east  = np.zeros([len(query_times), shape[0], shape[1]])
        wind_north = np.zeros([len(query_times), shape[0], shape[1]])
        for i in range(len(query_times)):
            wind_east[i,:]  = -wind_speed[i] * np.cos(-np.radians(wind_direction[i]) + np.pi/2)
            wind_north[i,:] = -wind_speed[i] * np.sin(-np.radians(wind_direction[i]) + np.pi/2)
        
        # Write
        np.save(prefix + "_wind_east.npy",  wind_east)
        np.save(prefix + "_wind_north.npy", wind_north)


def main():
    pass


if __name__ == "__main__":
    main()
