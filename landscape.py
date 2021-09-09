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
_LOHINUMVAL_TYPE = np.int32
_UNIT_OPTS_TYPE = np.int16
_DATA_TYPE = np.int16


def _parseLoHiNumVal(chunk):
    (lo, hi, num) = struct.unpack('iii', chunk[0:12])
    vals = np.array(array.array('i', chunk[12:412]), dtype=_LOHINUMVAL_TYPE)
    return (_LOHINUMVAL_TYPE(lo), _LOHINUMVAL_TYPE(hi), _LOHINUMVAL_TYPE(num), vals)


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


class Layer:
    def __init__(self):
        self.lo = _LOHINUMVAL_TYPE(0)
        self.hi = _LOHINUMVAL_TYPE(0)
        self.num = _LOHINUMVAL_TYPE(0)
        self.vals = np.zeros(_NUM_VALS, dtype=_LOHINUMVAL_TYPE)
        self.unit_opts = _UNIT_OPTS_TYPE(0)
        self.file = ""
        self.data = np.zeros([_NUM_NORTH_DEFAULT, _NUM_EAST_DEFAULT], dtype=_DATA_TYPE)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    @property
    def lo(self):
        return self._lo
    

    @lo.setter
    def lo(self, value):
        if not isinstance(value, _LOHINUMVAL_TYPE):
            raise TypeError("Layer.lo must be an instance of " + str(_LOHIHUMVAL_TYPE))
        self._lo = value
    

    @property
    def hi(self):
        return self._hi
    

    @hi.setter
    def hi(self, value):
        if not isinstance(value, _LOHINUMVAL_TYPE):
            raise TypeError("Layer.hi must be an instance of " + str(_LOHIHUMVAL_TYPE))
        self._hi = value
    

    @property
    def num(self):
        return self._num
    

    @num.setter
    def num(self, value):
        if not isinstance(value, _LOHINUMVAL_TYPE):
            raise TypeError("Layer.num must be an instance of " + str(_LOHIHUMVAL_TYPE))
        self._num = value
    

    @property
    def vals(self):
        return self._vals
    

    @vals.setter
    def vals(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Layer.vals must be an instance of " + str(np.ndarray))
        if value.shape != (_NUM_VALS,):
            raise ValueError("Layer.vals must have shape ({0},)".format(_NUM_VALS))
        if value.dtype != _LOHINUMVAL_TYPE:
            raise TypeError("Layer.vals must have dtype " + str(_LOHIHUMVAL_TYPE))
        self._vals = value
    

    @property
    def unit_opts(self):
        return self._unit_opts
    

    @unit_opts.setter
    def unit_opts(self, value):
        if not isinstance(value, _UNIT_OPTS_TYPE):
            raise TypeError("Layer.unit_opts must be an instance of " + str(_UNIT_OPTS_TYPE))
        self._unit_opts = value


    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Layer.data must be an np.ndarray")
        if value.ndim != 2:
            raise ValueError("Layer.data must have 2 dimensions")
        if value.dtype != _DATA_TYPE:
            raise TypeError("Layer.data must be an instance of " + str(_DATA_TYPE))

        self.lo = np.amin(value).astype(_LOHINUMVAL_TYPE)
        self.hi = np.amax(value).astype(_LOHINUMVAL_TYPE)
        self.vals = np.zeros(_NUM_VALS, dtype=_LOHINUMVAL_TYPE)
        vals_unique = np.unique(value)
        if len(vals_unique) > _NUM_VALS:
            self.num = _LOHINUMVAL_TYPE(-1)
        else:
            self.num = _LOHINUMVAL_TYPE(len(vals_unique))
            self.vals[0:self.num] = vals_unique.astype(_LOHINUMVAL_TYPE)
        self._data = value


    @property
    def shape(self):
        return self.data.shape


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
        self.layers = {'elevation': Layer(),
                       'slope':     Layer(),
                       'aspect':    Layer(),
                       'fuel':      Layer(),
                       'cover':     Layer(),
                       'height':    Layer(),
                       'base':      Layer(),
                       'density':   Layer(),
                       'duff':      Layer(),
                       'woody':     Layer()}

        if prefix:
            self.read(prefix)


    def __repr__(self):
        return "<Landscape description:%s>" % (self.description)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


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
            layer.data = np.zeros([self.num_north, self.num_east], dtype=_DATA_TYPE)

        (self.utm_east, self.utm_west, self.utm_north, self.utm_south) = struct.unpack('dddd', data[4172:4204])
        (self.units_grid) = struct.unpack('i', data[4204:4208])[0]
        (self.res_x, self.res_y) = struct.unpack('dd', data[4208:4224])

        unit_opts_arr = struct.unpack('hhhhhhhhhh', data[4224:4244])
        for i, layer in enumerate(self.layers.values()):
            layer.unit_opts = _UNIT_OPTS_TYPE(unit_opts_arr[i])

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
                self.layers['elevation'].data[i,j] = struct.unpack('h', data[data_i+0:data_i+ 2])[0]
                self.layers['slope'    ].data[i,j] = struct.unpack('h', data[data_i+2:data_i+ 4])[0]
                self.layers['aspect'   ].data[i,j] = struct.unpack('h', data[data_i+4:data_i+ 6])[0]
                self.layers['fuel'     ].data[i,j] = struct.unpack('h', data[data_i+6:data_i+ 8])[0]
                self.layers['cover'    ].data[i,j] = struct.unpack('h', data[data_i+8:data_i+10])[0]
                data_i += 10
                # Read crown fuel bands if present
                if self.crownPresent():
                    self.layers['height' ].data[i,j] = struct.unpack('h', data[data_i+0:data_i+2])[0]
                    self.layers['base'   ].data[i,j] = struct.unpack('h', data[data_i+2:data_i+4])[0]
                    self.layers['density'].data[i,j] = struct.unpack('h', data[data_i+4:data_i+6])[0]
                    data_i += 6
                # Read ground fuel bands if present
                if self.groundPresent():
                    self.layers['duff' ].data[i,j] = struct.unpack('h', data[data_i+0:data_i+2])[0]
                    self.layers['woody'].data[i,j] = struct.unpack('h', data[data_i+2:data_i+4])[0]
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
                file.write(struct.pack('i', self.layers['elevation'].data[i,j].astype(_DATA_TYPE)))
                file.write(struct.pack('i', self.layers['slope'    ].data[i,j].astype(_DATA_TYPE)))
                file.write(struct.pack('i', self.layers['aspect'   ].data[i,j].astype(_DATA_TYPE)))
                file.write(struct.pack('i', self.layers['fuel'     ].data[i,j].astype(_DATA_TYPE)))
                file.write(struct.pack('i', self.layers['cover'    ].data[i,j].astype(_DATA_TYPE)))
                # Write crown fuel bands if present
                if self.crownPresent():
                    file.write(struct.pack('i', self.layers['height' ].data[i,j].astype(_DATA_TYPE)))
                    file.write(struct.pack('i', self.layers['base'   ].data[i,j].astype(_DATA_TYPE)))
                    file.write(struct.pack('i', self.layers['density'].data[i,j].astype(_DATA_TYPE)))
                # Write ground fuel bands if present
                if self.groundPresent():
                    file.write(struct.pack('i', self.layers['duff' ].data[i,j].astype(_DATA_TYPE)))
                    file.write(struct.pack('i', self.layers['woody'].data[i,j].astype(_DATA_TYPE)))


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
            np.save(prefix + "_" + name, layer.data)


def main():
    pass


if __name__ == "__main__":
    main()
