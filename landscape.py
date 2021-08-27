import os
import argparse
import struct
import array
import numpy as np

_HEADER_LENGTH = 7316
_HILONUMVAL_LENGTH = 412
_FILE_LENGTH = 256

def _parseHiLoNumVal(chunk):
    (lo, hi, num) = struct.unpack('iii', chunk[0:12])
    vals = np.array(array.array('i', chunk[12:412]))
    return (lo, hi, num, vals)

def _parseString(chunk):
    (i,), chunk = struct.unpack("I", chunk[:4]), chunk[4:]
    s_raw, chunk = chunk[:i], chunk[i:]
    s_decoded = s_raw.decode('utf-8')
    return s_decoded.split('\x00', 1)[0]

class _Layer:
    def __init__(self, name):
        self.name = name
        self.lo = 0.0
        self.hi = 0.0
        self.num = 0
        self.vals = np.zeros(100, dtype=np.int32)
        self.unit_opts = 0
        self.file = ""
        self.value = np.zeros([100, 100], dtype=np.int16)
    
    def shape(self):
        return self.value.shape()
    
    def present(self):
        return self.num > 0


class Landscape:
    def __init__(self, filename=None):
        self.filename = filename
        self.layers = [Layer("elevation"),
                       Layer("slope"),
                       Layer("aspect"),
                       Layer("fuel"),
                       Layer("cover"),
                       Layer("height"),
                       Layer("base"),
                       Layer("density"),
                       Layer("duff"),
                       Layer("woody")]
        if self.filename is not None:
            self.readLCP(self.filename)


    def __parseHeader(self, data):
        (self.crown_fuels, self.ground_fuels, self.latitude) = struct.unpack('iii', data[0:12])
        (self.lo_east, self.hi_east, self.lo_north, self.hi_north) = struct.unpack('dddd', data[12:44])

        data_i = 44
        for layer in self.layers:
            (layer.lo, layer.hi, layer.num, layer.vals) = parseHiLoNumVal(data[data_i:data_i+_HILONUMVAL_LENGTH])
            data_i += _HILONUMVAL_LENGTH

        (self.num_east, self.num_north) = struct.unpack('ii', data[4164:4172])
        (self.utm_east, self.utm_west, self.utm_north, self.utm_south) = struct.unpack('dddd', data[4172:4204])
        (self.units_grid) = struct.unpack('i', data[4204:4208])
        (self.res_x, self.res_y) = struct.unpack('dd', data[4208:4224])

        unit_opts_arr = struct.unpack('hhhhhhhhhh', data[4224:4244])
        for (i, layer) in enumerate(self.layers):
            layer.unit_opts = unit_opts_arr[i]

        data_i = 4244
        for layer in self.layers:
            layer.file = parseString(data[data_i:data_i+_FILE_LENGTH])
            data_i += _FILE_LENGTH

        self.description = parseString(data[6804:7316])


    def __parseBody(self, data):
        data_i = 0
        for i in range(self.num_east):
            for j in range(self.num_north):
                for layer in self.layers:
                    if layer.num > 0:
                        layer.value[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                        data_i += 2


    def readLCP(self, filename):
        with open(filename, "rb") as file:
            file_data = file.read()

            header = file_data[0:7316]
            body = file_data[7316:]

            self.parseHeader(header)
            self.parseBody(body)
    

    def __writeHeader(self, filename):
        raise NotImplementedError
    

    def __writeBody(self, filename):
        raise NotImplementedError


    def writeLCP(self, filename):
        self.writeHeader(filename)
        self.writeBody(filename)
    

    def writeNPY(self, filename):
        raise NotImplementedError


def main():
    filename = "./flat_crown.lcp"
    landscape = Landscape(filename)

    import code; code.interact(local=locals())

    print(landscape.description)

if __name__ == "__main__":
    main()