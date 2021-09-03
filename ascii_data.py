"""Class for managing standard GIS ascii files"""

import numpy as np

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
    

    def __parseHeaderLine(self, line):
        return line.strip().split(" ")
    

    def __readHeader(self, filename):
        with open(filename, "r") as file:
            self.nrows = int(self.__parseHeaderLine(file.readline())[1])
            self.ncols = int(self.__parseHeaderLine(file.readline())[1])
            self.xllcorner = float(self.__parseHeaderLine(file.readline())[1])
            self.yllcorner = float(self.__parseHeaderLine(file.readline())[1])
            self.cell_size = float(self.__parseHeaderLine(file.readline())[1])
            self.nodata_value = float(self.__parseHeaderLine(file.readline())[1])
    

    def __readBody(self, filename):
        self.data = np.loadtxt(filename, skiprows=6, dtype=np.float64)
        self.data = np.ma.masked_where(
            np.isclose(self.data, self.nodata_value),
            self.data)


    def read(self, filename):
        self.__readHeader(filename)
        self.__readBody(filename)
    

    def __writeHeader(self, filename):
        raise NotImplementedError
    

    def __writeBody(self, filename):
        raise NotImplementedError
    

    def write(self, filename):
        self.__writeHeader(filename)
        self.__writeBody(filename)
        raise NotImplementedError

    
    def writeNPY(self, filename):
        raise NotImplementedError


def main():
    pass


if __name__ == "__main__":
    main()
