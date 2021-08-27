import os
import warnings
import struct
import array
import numpy as np

_HEADER_LENGTH = 7316
_LOHINUMVAL_LENGTH = 412
_FILE_LENGTH = 256
_DESCRIPTION_LENGTH = 512

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


    def __parseHeader(self, data):
        (self.crown_fuels, self.ground_fuels, self.latitude) = struct.unpack('iii', data[0:12])
        (self.lo_east, self.hi_east, self.lo_north, self.hi_north) = struct.unpack('dddd', data[12:44])

        data_i = 44
        for (i, layer) in enumerate(self.layers):
            (layer.lo, layer.hi, layer.num, layer.vals) = _parseLoHiNumVal(data[data_i:data_i+_LOHINUMVAL_LENGTH])
            if (i < 5) and layer.num == 0:
                warnings.warn("No data for mandatory layer " + layer.name)
            data_i += _LOHINUMVAL_LENGTH

        (self.num_east, self.num_north) = struct.unpack('ii', data[4164:4172])
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
        for i in range(self.num_east):
            for j in range(self.num_north):
                for layer in self.layers:
                    if layer.num > 0:
                        layer.value[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
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
            if (i < 5) and layer.num == 0:
                raise IOError("No data for mandatory layer " + layer.name)
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
        for i in range(self.num_east):
            for j in range(self.num_north):
                for layer in self.layers:
                    if layer.num > 0:
                        file.write(struct.pack('h', layer.value[i,j]))


    def writeLCP(self, filename):
        with open(filename, "wb") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
    

    def writeNPY(self, filename):
        raise NotImplementedError


def main():
    filename = "./flat_nocrown.lcp"
    landscape = Landscape(filename)
    landscape.writeLCP("test.lcp")

    import code; code.interact(local=locals())

    print(landscape.description)

if __name__ == "__main__":
    main()