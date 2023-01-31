"""Utilities for reading and writing ATM files and their associated wind files."""

import os
from enum import Enum
import datetime as dt
import numpy as np
import pandas as pd
from collections import Counter

from . import ascii_data


_HEADER_LINES = 3
_DEFAULT_YEAR = 2000
_DATA_COLS = [
    'time',
    'wind_speed_file',
    'wind_speed_data',
    'wind_direction_file',
    'wind_direction_data']


class Type(Enum):
    WINDS_AND_CLOUDS = 1
    WEATHER_AND_WINDS = 2


class Unit(Enum):
    ENGLISH = 1
    METRIC = 2


class ATM:
    def __init__(self, filename=None):
        self.type = Type.WINDS_AND_CLOUDS
        self.units = Unit.ENGLISH
        self.data = pd.DataFrame(columns = _DATA_COLS)
        self.root_dir = "./"

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
    

    @property
    def count(self):
        return len(self.data)
    

    @property
    def shape(self):
        return self.data.loc[0, 'wind_speed_data'].shape
    

    def __parseHeader(self, file):
        # Parse elevation
        line1 = file.readline()
        try:
            self.type = Type[line1.strip().upper()]
        except KeyError:
            raise IOError("Invalid type specification. Must be WINDS_AND_CLOUDS or WEATHER_AND_WINDS.")

        # Parse unit system
        line2 = file.readline()
        try:
            self.units = Unit[line2.strip().upper()]
        except KeyError:
            raise IOError("Invalid units specification. Must be ENGLISH or METRIC.")
    

    def __parseBodyLine(self, line):
        vals = line.strip().split()
        entry = {}
        entry['time'] = dt.datetime(
            _DEFAULT_YEAR,
            int(vals[0]),
            int(vals[1]),
            int("{0:04d}".format(int(vals[2]))[0:2]),
            int("{0:04d}".format(int(vals[2]))[2:4]))
        entry['wind_speed_file'] =     vals[3]
        entry['wind_direction_file'] = vals[4]
        entry['wind_speed_data'] =     ascii_data.ASCIIData(os.path.join(self.root_dir, vals[3]))
        entry['wind_direction_data'] = ascii_data.ASCIIData(os.path.join(self.root_dir, vals[4]))
        return entry
    

    def __parseBody(self, file):
        for line in file:
            entry = self.__parseBodyLine(line)
            self.data = self.data.append(entry, ignore_index=True)
    

    def read(self, filename):
        self.root_dir = os.path.split(filename)[0]
        self.data = self.data[0:0]
        with open(filename, "r") as file:
            self.__parseHeader(file)
            self.__parseBody(file)
    

    def __writeHeader(self, file):
        file.write(self.type.name.upper() + "\n")
        file.write(self.units.name.upper() + "\n")
    

    def __writeBodyLine(self, file, entry):
        wind_dir_local = os.path.normpath(self.root_dir).split(os.sep)[-1]
        file.write("{0:02d} {1:02d} {2:02d}{3:02d} {4} {5}\n".format(
            entry['time'].month,
            entry['time'].day,
            entry['time'].hour,
            entry['time'].minute,
            os.path.join(wind_dir_local, entry['wind_speed_file']),
            os.path.join(wind_dir_local, entry['wind_direction_file'])))
    

    def __writeBody(self, file):
        for i in range(self.count):
            self.__writeBodyLine(file, self.data.iloc[i])
    

    def __writeFilesLine(self, entry):
        entry['wind_speed_data'    ].write(os.path.join(self.root_dir, entry['wind_speed_file'    ]))
        entry['wind_direction_data'].write(os.path.join(self.root_dir, entry['wind_direction_file']))
    

    def __writeFiles(self):
        for i in range(self.count):
            self.__writeFilesLine(self.data.loc[i])
    
    
    def write(self, filename):
        with open(filename, "w") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
        self.__writeFiles()
    

    def writeNPY(self, prefix, query_times=None):
        if not query_times:
            query_times = [self.data.loc[0, 'time']]
        
        #TODO: spatial interpolation when wind grid and landscape grid are not matched
        
        # Convert datetimes to elapsed seconds
        epoch = self.data.loc[0, 'time'].replace(year=query_times[0].year)
        query_times_secs = [(time - epoch).total_seconds() for time in query_times]
        data_times_secs = \
            [(time.replace(year=query_times[0].year).to_pydatetime() - epoch).total_seconds() for time in self.data['time']]

        wind_npy_shape = [len(query_times), self.shape[0], self.shape[1]]

        # Interpolate data at query times (previous)
        wind_speed     = np.zeros(wind_npy_shape)
        wind_direction = np.zeros(wind_npy_shape)
        for i in range(len(query_times)):
            i_data = np.argwhere(query_times_secs[i] >= np.array(data_times_secs))[-1][0]
            wind_speed[i]     = self.data.loc[i_data, 'wind_speed_data'    ].data
            wind_direction[i] = self.data.loc[i_data, 'wind_direction_data'].data

        # Compute components from speed and direction
        wind_east  = np.zeros(wind_npy_shape)
        wind_north = np.zeros(wind_npy_shape)
        for i in range(len(query_times)):
            wind_east[i,:]  = -wind_speed[i] * np.cos(-np.radians(wind_direction[i]) + np.pi/2)
            wind_north[i,:] = -wind_speed[i] * np.sin(-np.radians(wind_direction[i]) + np.pi/2)
        
        # Write
        np.save(prefix + "_wind_east.npy",  wind_east)
        np.save(prefix + "_wind_north.npy", wind_north)

        import code; code.interact(local=locals())


def main():
    pass


if __name__ == "__main__":
    main()
