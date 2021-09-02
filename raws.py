"""Utilities for reading and writing RAWS files."""
import datetime as dt
import pandas as pd


_HEADER_LINES = 3


class RAWS:
    def __init__(self, filename=None):
        self.filename = filename
        self.elevation = 0
        self.units = "ENGLISH"
        self.count = 0
        self.data = pd.DataFrame(
            columns = ["time",
                       "temperature",
                       "humidity",
                       "precipitation",
                       "wind_speed",
                       "wind_direction",
                       "cloud_cover"]
        )

        if filename:
            self.read(filename)
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

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
            units = val2.upper()
            if units != "ENGLISH":
                raise IOError("RAWS files only support English units")
            else:
                self.units = units
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
            int(vals[3][0:2]),
            int(vals[3][2:4])
        )
        entry['temperature'] =    int(vals[4])
        entry['humidity'] =       float(vals[5]) / 100
        entry['precipitation'] =  float(vals[6])
        entry['wind_speed'] =     int(vals[7])
        entry['wind_direction'] = int(vals[8])
        entry['cloud_cover'] =    float(vals[9]) / 100
        return entry
    

    def __parseBody(self, file):
        for i in range(self.count):
            line = file.readline()
            entry = self.__parseBodyLine(line)
            self.data = self.data.append(entry, ignore_index=True)


    def read(self, filename):
        with open(filename, "r") as file:
            self.__parseHeader(file)
            self.__parseBody(file)
    

    def __writeHeader(self, file):
        file.write("RAWS_ELEVATION: {0}\n".format(self.elevation))
        file.write("RAWS_UNITS: " + self.units.title() + "\n")
        file.write("RAWS: {0}\n".format(self.count))
    

    def __writeBodyLine(self, file, entry):
        file.write("{0} {1} {2} {3}{4:02d} {5} {6} {7:1.2f} {8} {9} {10}\n".format(
            entry['time'].year,
            entry['time'].month,
            entry['time'].day,
            entry['time'].hour,
            entry['time'].minute,
            entry['temperature'],
            int(entry['humidity'] * 100),
            entry['precipitation'],
            entry['wind_speed'],
            entry['wind_direction'],
            int(entry['cloud_cover'] * 100)
        ))
    

    def __writeBody(self, file):
        for i in range(self.count):
            self.__writeBodyLine(file, self.data.loc[i])
    
    
    def write(self, filename):
        with open(filename, "w") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
