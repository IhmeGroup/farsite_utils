import os
import datetime as dt
import numpy as np
import pandas as pd
from collections import Counter

from . import ascii_data

_DEFAULT_YEAR = 2000
_DATA_COLS = [
    'time',
    'wind_speed',
    'wind_direction']

class ATM:
    def __init__(self, filename=None):
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
    

    def __parseBodyLine(self, line):
        vals = line.strip().split()
        entry = {}
        entry['time'] = dt.datetime(
            _DEFAULT_YEAR,
            int(vals[0]),
            int(vals[1]),
            int("{0:04d}".format(int(vals[2]))[0:2]),
            int("{0:04d}".format(int(vals[2]))[2:4]))
        entry['wind_speed'] =     ascii_data.ASCIIData(os.path.join(self.root_dir, vals[3]))
        entry['wind_direction'] = ascii_data.ASCIIData(os.path.join(self.root_dir, vals[4]))
        return entry
    

    def __parseBody(self, file):
        for line in file:
            entry = self.__parseBodyLine(line)
            self.data = self.data.append(entry, ignore_index=True)
    

    def read(self, filename):
        self.root_dir = os.path.split(filename)[0]
        self.data = self.data[0:0]
        with open(filename, "r") as file:
            self.__parseBody(file)
