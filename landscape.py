"""Utitities for reading, writing, and converting landscape files."""
import struct
import array
import numpy as np
from osgeo import osr


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
    def __init__(self):
        self.lo = 0.0
        self.hi = 0.0
        self.num = 0
        self.vals = np.zeros(_NUM_VALS, dtype=np.int32)
        self.unit_opts = 0
        self.file = ""
        self._value = np.zeros([_NUM_NORTH_DEFAULT, _NUM_EAST_DEFAULT], dtype=np.int16)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


    @property
    def shape(self):
        return self.value.shape


    @property
    def value(self):
        return self._value
    

    @value.setter
    def value(self, given_value):
        self.lo = np.amin(given_value)
        self.hi = np.amax(given_value)
        self.vals = np.zeros(_NUM_VALS, dtype=np.int32)
        vals_unique = np.unique(given_value)
        if len(vals_unique) > 100:
            self.num = -1
        else:
            self.num = len(vals_unique)
            self.vals[0:self.num] = vals_unique
        self._value = given_value


class Landscape:
    def __init__(self, prefix=None):
        self.srs = osr.SpatialReference()
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
        self.layers = {'elevation': _Layer(),
                       'slope':     _Layer(),
                       'aspect':    _Layer(),
                       'fuel':      _Layer(),
                       'cover':     _Layer(),
                       'height':    _Layer(),
                       'base':      _Layer(),
                       'density':   _Layer(),
                       'duff':      _Layer(),
                       'woody':     _Layer()}

        if prefix:
            self.read(prefix)


    def __repr__(self):
        return "<Landscape description:%s>" % (self.description)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    @property
    def shape(self):
        return self.value.shape


    @property
    def center(self):
        return ((self.utm_west  + self.utm_east)  / 2.0,
                (self.utm_south + self.utm_north) / 2.0)


    @property
    def size(self):
        return ((self.utm_east -  self.utm_west),
                (self.utm_north - self.utm_south))


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
        for layer in self.layers.values():
            (layer.lo, layer.hi, layer.num, layer.vals) = _parseLoHiNumVal(data[data_i:data_i+_LOHINUMVAL_LENGTH])
            data_i += _LOHINUMVAL_LENGTH

        (self.num_east, self.num_north) = struct.unpack('ii', data[4164:4172])
        for layer in self.layers.values():
            layer.value = np.zeros([self.num_north, self.num_east], dtype=np.int16)

        (self.utm_east, self.utm_west, self.utm_north, self.utm_south) = struct.unpack('dddd', data[4172:4204])
        (self.units_grid) = struct.unpack('i', data[4204:4208])[0]
        (self.res_x, self.res_y) = struct.unpack('dd', data[4208:4224])

        unit_opts_arr = struct.unpack('hhhhhhhhhh', data[4224:4244])
        for i, layer in enumerate(self.layers.values()):
            layer.unit_opts = unit_opts_arr[i]

        data_i = 4244
        for layer in self.layers.values():
            layer.file = _parseEncodedString(data[data_i:data_i+_FILE_LENGTH])
            data_i += _FILE_LENGTH

        self.description = _parseEncodedString(data[6804:7316])


    def __parseBody(self, data):
        data_i = 0
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Read required bands
                self.layers['elevation'].value[i,j] = struct.unpack('h', data[data_i+0:data_i+ 2])[0]
                self.layers['slope'    ].value[i,j] = struct.unpack('h', data[data_i+2:data_i+ 4])[0]
                self.layers['aspect'   ].value[i,j] = struct.unpack('h', data[data_i+4:data_i+ 6])[0]
                self.layers['fuel'     ].value[i,j] = struct.unpack('h', data[data_i+6:data_i+ 8])[0]
                self.layers['cover'    ].value[i,j] = struct.unpack('h', data[data_i+8:data_i+10])[0]
                data_i += 10
                # Read crown fuel bands if present
                if self.crownPresent():
                    self.layers['height' ].value[i,j] = struct.unpack('h', data[data_i+0:data_i+2])[0]
                    self.layers['base'   ].value[i,j] = struct.unpack('h', data[data_i+2:data_i+4])[0]
                    self.layers['density'].value[i,j] = struct.unpack('h', data[data_i+4:data_i+6])[0]
                    data_i += 6
                # Read ground fuel bands if present
                if self.groundPresent():
                    self.layers['duff' ].value[i,j] = struct.unpack('h', data[data_i+0:data_i+2])[0]
                    self.layers['woody'].value[i,j] = struct.unpack('h', data[data_i+2:data_i+4])[0]
                    data_i += 4


    def readLCP(self, filename):
        with open(filename, "rb") as file:
            file_data = file.read()

            header = file_data[0:_HEADER_LENGTH]
            body = file_data[_HEADER_LENGTH:]

            self.__parseHeader(header)
            self.__parseBody(body)
    

    def readProjection(self, filename):
        with open(filename, "r") as file:
            prj_txt = file.read()
        self.srs.ImportFromESRI([prj_txt])
        self.srs.AutoIdentifyEPSG()
    

    def writeProjection(self, filename):
        with open(filename, "w") as file:
            file.write(self.srs.ExportToWkt())
    

    def read(self, prefix):
        self.readLCP(prefix + ".lcp")
        self.readProjection(prefix + ".prj")


    def __writeHeader(self, file):
        file.write(struct.pack('iii', self.crown_fuels, self.ground_fuels, self.latitude))
        file.write(struct.pack('dddd', self.lo_east, self.hi_east, self.lo_north, self.hi_north))

        for layer in self.layers.values():
            file.write(_buildLoHiNumVal(layer.lo, layer.hi, layer.num, layer.vals))
        
        file.write(struct.pack('ii', self.num_east, self.num_north))
        file.write(struct.pack('dddd', self.utm_east, self.utm_west, self.utm_north, self.utm_south))
        file.write(struct.pack('i', self.units_grid))
        file.write(struct.pack('dd', self.res_x, self.res_y))

        unit_opts_arr = [layer.unit_opts for layer in self.layers.values()]
        file.write(struct.pack('hhhhhhhhhh', *unit_opts_arr))

        for layer in self.layers.values():
            file.write(_buildEncodedString(layer.file, _FILE_LENGTH))
        
        file.write(_buildEncodedString(self.description, _DESCRIPTION_LENGTH))

        if file.tell() != _HEADER_LENGTH:
            raise IOError("Issue writing header...")


    def __writeBody(self, file):
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Write required bands
                file.write(struct.pack('h', self.layers['elevation'].value[i,j]))
                file.write(struct.pack('h', self.layers['slope'    ].value[i,j]))
                file.write(struct.pack('h', self.layers['aspect'   ].value[i,j]))
                file.write(struct.pack('h', self.layers['fuel'     ].value[i,j]))
                file.write(struct.pack('h', self.layers['cover'    ].value[i,j]))
                # Write crown fuel bands if present
                if self.crownPresent():
                    file.write(struct.pack('h', self.layers['height' ].value[i,j]))
                    file.write(struct.pack('h', self.layers['base'   ].value[i,j]))
                    file.write(struct.pack('h', self.layers['density'].value[i,j]))
                # Write ground fuel bands if present
                if self.groundPresent():
                    file.write(struct.pack('h', self.layers['duff' ].value[i,j]))
                    file.write(struct.pack('h', self.layers['woody'].value[i,j]))


    def writeLCP(self, filename):
        with open(filename, "wb") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
    

    def write(self, prefix):
        self.writeLCP(prefix+".lcp")
        self.writeProjection(prefix+".prj")
    

    def projection(self, format):
        if format == "WKT":
            return str(self.srs.ExportToWkt())
        elif format == "PROJ4":
            return str(self.srs.ExportToProj4())
        elif format == "EPSG":
            return str(self.srs.GetAuthorityCode(None))


    def writeNPY(self, prefix):
        for name, layer in self.layers.items():
            np.save(prefix + "_" + name, layer.value)


def main():
    pass


if __name__ == "__main__":
    main()
