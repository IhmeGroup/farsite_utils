"""Class for managing standard GIS ascii files"""

import os
import numpy as np

_PRECISION = 8

class ASCIIData:
    def __init__(self, filename=None):
        self.nrows = 0
        self.ncols = 0
        self.xllcorner = 0.0
        self.yllcorner = 0.0
        self.cell_size = 30.0
        self.nodata_value = -9999.0
        self.data = np.zeros([self.nrows, self.ncols])

        if filename:
            self.read(filename)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    @property
    def shape(self):
        return (self.nrows, self.ncols)
    

    def __parseHeaderLine(self, line):
        return line.strip().split()
    

    def __readHeader(self, file):
        self.ncols = int(self.__parseHeaderLine(file.readline())[1])
        self.nrows = int(self.__parseHeaderLine(file.readline())[1])
        self.xllcorner = float(self.__parseHeaderLine(file.readline())[1])
        self.yllcorner = float(self.__parseHeaderLine(file.readline())[1])
        self.cell_size = float(self.__parseHeaderLine(file.readline())[1])
        self.nodata_value = float(self.__parseHeaderLine(file.readline())[1])
    

    def __readBody(self, file):
        self.data = np.loadtxt(file, dtype=np.float64)
        self.data = np.ma.masked_where(
            np.isclose(self.data, self.nodata_value),
            self.data)


    def read(self, filename):
        with open(filename, "r") as file:
            self.__readHeader(file)
            self.__readBody(file)
    

    def __writeHeader(self, file):
        file.write("NCOLS: {0}\n".format(self.ncols))
        file.write("NROWS: {0}\n".format(self.nrows))
        file.write("XLLCORNER: {0}\n".format(self.xllcorner))
        file.write("YLLCORNER: {0}\n".format(self.yllcorner))
        file.write("CELLSIZE: {0}\n".format(self.cell_size))
        file.write("NODATA_VALUE: {0}\n".format(self.nodata_value))
    

    def __writeBody(self, file):
        np.savetxt(file, self.data, fmt='%.{0}f'.format(_PRECISION))
    

    def write(self, filename):
        root_dir = os.path.split(filename)[0]
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        with open(filename, "w") as file:
            self.__writeHeader(file)
            self.__writeBody(file)

    
    def writeNPY(self, filename):
        raise NotImplementedError


def main():
    pass


if __name__ == "__main__":
    main()
