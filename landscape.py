"""Utitities for reading, writing, and converting landscape files."""

import os
import warnings
import struct
import array
import numpy as np


_HEADER_LENGTH = 7316
_LOHINUMVAL_LENGTH = 412
_FILE_LENGTH = 256
_DESCRIPTION_LENGTH = 512
_NUM_VALS = 100
_NUM_EAST_DEFAULT = 100
_NUM_NORTH_DEFAULT = 100


def _parseLoHiNumVal(chunk):
    (lo, hi, num) = struct.unpack('iii', chunk[0:12])
    vals = np.array(array.array('i', chunk[12:412]))
    return (lo, hi, num, vals)


def _parseEncodedString(chunk):
    decoded = chunk.decode('utf-8')
    return decoded.split('\x00', 1)[0]


def _buildLoHiNumVal(lo, hi, num, vals):
    chunk = struct.pack("iii", lo, hi, num)
    chunk += bytearray(vals)
    return chunk


def _buildEncodedString(s, length):
    padded = "{0:\x00<{1}}".format(s, length)
    return padded.encode('utf-8')


class _Layer:
    def __init__(self, name):
        self.name = name
        self.lo = 0.0
        self.hi = 0.0
        self.num = 0
        self.vals = np.zeros(_NUM_VALS, dtype=np.int32)
        self.unit_opts = 0
        self.file = ""
        self.value = np.zeros([_NUM_NORTH_DEFAULT, _NUM_EAST_DEFAULT], dtype=np.int16)


    def __repr__(self):
        return "<Layer name:%s>" % (self.name)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


    def shape(self):
        return self.value.shape()


class Landscape:
    def __init__(self, filename=None):
        self.filename = filename
        self.crown_fuels = 20
        self.ground_fuels = 20
        self.latitude = 0
        self.lo_east = 0
        self.hi_east = 0
        self.lo_north = 0
        self.hi_north = 0
        self.num_east = _NUM_EAST_DEFAULT
        self.num_north = _NUM_NORTH_DEFAULT
        self.utm_east = 0.0
        self.utm_west = 0.0
        self.utm_north = 0.0
        self.utm_south = 0.0
        self.units_grid = 0
        self.res_x = 0.0
        self.res_y = 0.0
        self.description = ""
        self.layers = [_Layer("elevation"),
                       _Layer("slope"),
                       _Layer("aspect"),
                       _Layer("fuel"),
                       _Layer("cover"),
                       _Layer("height"),
                       _Layer("base"),
                       _Layer("density"),
                       _Layer("duff"),
                       _Layer("woody")]

        if self.filename is not None:
            self.readLCP(self.filename)


    def __repr__(self):
        return "<Landscape description:%s>" % (self.description)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


    def crownPresent(self):
        if self.crown_fuels == 20:
            return False
        elif self.crown_fuels == 21:
            return True
        else:
            raise ValueError("crown_fuels has invalid value: {0}".format(self.crown_fuels))


    def groundPresent(self):
        if self.ground_fuels == 20:
            return False
        elif self.ground_fuels == 21:
            return True
        else:
            raise ValueError("ground_fuels has invalid value: {0}".format(self.ground_fuels))


    def __parseHeader(self, data):
        (self.crown_fuels, self.ground_fuels, self.latitude) = struct.unpack('iii', data[0:12])
        (self.lo_east, self.hi_east, self.lo_north, self.hi_north) = struct.unpack('dddd', data[12:44])

        data_i = 44
        for (i, layer) in enumerate(self.layers):
            (layer.lo, layer.hi, layer.num, layer.vals) = _parseLoHiNumVal(data[data_i:data_i+_LOHINUMVAL_LENGTH])
            data_i += _LOHINUMVAL_LENGTH

        (self.num_east, self.num_north) = struct.unpack('ii', data[4164:4172])
        for layer in self.layers:
            layer.value = np.zeros([self.num_north, self.num_east], dtype=np.int16)

        (self.utm_east, self.utm_west, self.utm_north, self.utm_south) = struct.unpack('dddd', data[4172:4204])
        (self.units_grid) = struct.unpack('i', data[4204:4208])[0]
        (self.res_x, self.res_y) = struct.unpack('dd', data[4208:4224])

        unit_opts_arr = struct.unpack('hhhhhhhhhh', data[4224:4244])
        for (i, layer) in enumerate(self.layers):
            layer.unit_opts = unit_opts_arr[i]

        data_i = 4244
        for layer in self.layers:
            layer.file = _parseEncodedString(data[data_i:data_i+_FILE_LENGTH])
            data_i += _FILE_LENGTH

        self.description = _parseEncodedString(data[6804:7316])


    def __parseBody(self, data):
        data_i = 0
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Read required bands
                for k in range(0, 5):
                    self.layers[k].value[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                    data_i += 2
                # Read crown fuel bands if present
                if self.crownPresent():
                    for k in range(5, 8):
                        self.layers[k].value[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                        data_i += 2
                # Read ground fuel bands if present
                if self.groundPresent():
                    for k in range(8, 10):
                        self.layers[k].value[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                        data_i += 2


    def readLCP(self, filename):
        with open(filename, "rb") as file:
            file_data = file.read()

            header = file_data[0:_HEADER_LENGTH]
            body = file_data[_HEADER_LENGTH:]

            self.__parseHeader(header)
            self.__parseBody(body)


    def __writeHeader(self, file):
        file.write(struct.pack('iii', self.crown_fuels, self.ground_fuels, self.latitude))
        file.write(struct.pack('dddd', self.lo_east, self.hi_east, self.lo_north, self.hi_north))

        for (i, layer) in enumerate(self.layers):
            file.write(_buildLoHiNumVal(layer.lo, layer.hi, layer.num, layer.vals))
        
        file.write(struct.pack('ii', self.num_east, self.num_north))
        file.write(struct.pack('dddd', self.utm_east, self.utm_west, self.utm_north, self.utm_south))
        file.write(struct.pack('i', self.units_grid))
        file.write(struct.pack('dd', self.res_x, self.res_y))

        unit_opts_arr = [layer.unit_opts for layer in self.layers]
        file.write(struct.pack('hhhhhhhhhh', *unit_opts_arr))

        for layer in self.layers:
            file.write(_buildEncodedString(layer.file, _FILE_LENGTH))
        
        file.write(_buildEncodedString(self.description, _DESCRIPTION_LENGTH))

        if file.tell() != _HEADER_LENGTH:
            raise IOError("Issue writing header...")


    def __writeBody(self, file):
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Write required bands
                for k in range(0, 5):
                    file.write(struct.pack('h', self.layers[k].value[i,j]))
                # Write crown fuel bands if present
                if self.crownPresent():
                    for k in range(5, 8):
                        file.write(struct.pack('h', self.layers[k].value[i,j]))
                # Write ground fuel bands if present
                if self.groundPresent():
                    for k in range(8, 10):
                        file.write(struct.pack('h', self.layers[k].value[i,j]))


    def writeLCP(self, filename):
        with open(filename, "wb") as file:
            self.__writeHeader(file)
            self.__writeBody(file)


    def writeNPY(self, prefix):
        for layer in self.layers:
            np.save(prefix + "_" + layer.name, layer.value)


def main():
    pass


if __name__ == "__main__":
    main()
