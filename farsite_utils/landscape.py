"""Utitities for reading, writing, and converting landscape files."""
import struct
import array
import numpy as np


_HEADER_LENGTH = 7316
_LOHINUMVAL_LENGTH = 412
_FILE_STRING_LENGTH = 256
_DESCRIPTION_LENGTH = 512
_NUM_VALS = 100
_NUM_EAST_DEFAULT = 100
_NUM_NORTH_DEFAULT = 100
_LOHINUMVAL_TYPE = np.int32
_UNIT_OPTS_TYPE = np.int16
_DATA_TYPE = np.int16
_LAYER_NAMES_REQUIRED = ['elevation', 'slope', 'aspect', 'fuel', 'cover']
_LAYER_NAMES_CROWN = ['height', 'base', 'density']
_LAYER_NAMES_GROUND = ['duff', 'woody']
_LAYER_NAMES = _LAYER_NAMES_REQUIRED + _LAYER_NAMES_CROWN + _LAYER_NAMES_GROUND
_NODATA_VALUE = -9999


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
    def __init__(self, shape):
        self.lo = _LOHINUMVAL_TYPE(0)
        self.hi = _LOHINUMVAL_TYPE(0)
        self.num = _LOHINUMVAL_TYPE(0)
        self.vals = np.zeros(_NUM_VALS, dtype=_LOHINUMVAL_TYPE)
        self.unit_opts = _UNIT_OPTS_TYPE(0)
        self.file = ""
        if shape:
            self.data = np.zeros(shape, dtype=_DATA_TYPE)
        else:
            self.data = np.zeros([_NUM_NORTH_DEFAULT, _NUM_EAST_DEFAULT], dtype=_DATA_TYPE)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def __eq__(self, other):
        if type(other) is type(self):
            return (
                self.lo == other.lo and
                self.hi == other.hi and
                self.num == other.num and
                np.array_equal(self.vals, other.vals) and
                np.array_equal(self.data, other.data))
        return False
    

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
        elif len(vals_unique) == _NUM_VALS and np.any(vals_unique == _NODATA_VALUE):
            self.num = _LOHINUMVAL_TYPE(-1)
        elif len(vals_unique) == _NUM_VALS and not np.any(vals_unique == 0):
            self.num = _LOHINUMVAL_TYPE(-1)
        else:
            if vals_unique[0] != 0:
                vals_unique = np.append(0, vals_unique)
            self.num = _LOHINUMVAL_TYPE(len(vals_unique))
            self.vals[0:self.num] = vals_unique.astype(_LOHINUMVAL_TYPE)
        self._data = value


    @property
    def shape(self):
        return self.data.shape


class Landscape:
    def __init__(self, prefix=None, shape=None):
        self.projection = ""
        self.crown_fuels = 20
        self.ground_fuels = 20
        self.latitude = 0
        self.lo_east = 0
        self.hi_east = 0
        self.lo_north = 0
        self.hi_north = 0
        if shape:
            self.num_north = shape[0]
            self.num_east = shape[1]
        else:
            self.num_north = _NUM_NORTH_DEFAULT
            self.num_east = _NUM_EAST_DEFAULT
        self.utm_east = 0.0
        self.utm_west = 0.0
        self.utm_north = 0.0
        self.utm_south = 0.0
        self.units_grid = 0
        self.res_x = 0.0
        self.res_y = 0.0
        self.description = ""
        self.layers = {name: Layer(shape) for name in _LAYER_NAMES}

        if prefix:
            self.read(prefix)


    def __repr__(self):
        return "<Landscape description:%s>" % (self.description)


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False


    @property
    def center(self):
        return ((self.utm_west  + self.utm_east)  / 2.0,
                (self.utm_south + self.utm_north) / 2.0)


    @property
    def size(self):
        return ((self.utm_east -  self.utm_west),
                (self.utm_north - self.utm_south))
    

    @property
    def shape(self):
        return (self.num_north, self.num_east)
    

    @property
    def area(self):
        return self.size[0] * self.size[1]
    

    @shape.setter
    def shape(self, value):
        self.num_north = value[0]
        self.num_east  = value[1]


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
            layer._data = np.zeros([self.num_north, self.num_east], dtype=_DATA_TYPE)

        (self.utm_east, self.utm_west, self.utm_north, self.utm_south) = struct.unpack('dddd', data[4172:4204])
        (self.units_grid) = struct.unpack('i', data[4204:4208])[0]
        (self.res_x, self.res_y) = struct.unpack('dd', data[4208:4224])

        unit_opts_arr = struct.unpack('hhhhhhhhhh', data[4224:4244])
        for i, layer in enumerate(self.layers.values()):
            layer.unit_opts = _UNIT_OPTS_TYPE(unit_opts_arr[i])

        data_i = 4244
        for layer in self.layers.values():
            layer.file = _parseEncodedString(data[data_i:data_i+_FILE_STRING_LENGTH])
            data_i += _FILE_STRING_LENGTH

        self.description = _parseEncodedString(data[6804:7316])


    def __parseBody(self, data):
        data_i = 0
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Read required bands
                for name in _LAYER_NAMES_REQUIRED:
                    self.layers[name].data[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                    data_i += 2
                # Read crown fuel bands if present
                if self.crownPresent():
                    for name in _LAYER_NAMES_CROWN:
                        self.layers[name].data[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                        data_i += 2
                # Read ground fuel bands if present
                if self.groundPresent():
                    for name in _LAYER_NAMES_GROUND:
                        self.layers[name].data[i,j] = struct.unpack('h', data[data_i:data_i+2])[0]
                        data_i += 2


    def readLCP(self, filename):
        with open(filename, "rb") as file:
            file_data = file.read()

            header = file_data[0:_HEADER_LENGTH]
            body = file_data[_HEADER_LENGTH:]

            self.__parseHeader(header)
            self.__parseBody(body)
    

    def readProjection(self, filename):
        with open(filename, "r") as file:
            self.projection = file.read()
    

    def writeProjection(self, filename):
        with open(filename, "w") as file:
            file.write(self.projection)
    

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
            file.write(_buildEncodedString(layer.file, _FILE_STRING_LENGTH))
        
        file.write(_buildEncodedString(self.description, _DESCRIPTION_LENGTH))

        if file.tell() != _HEADER_LENGTH:
            raise IOError("Issue writing header...")


    def __writeBody(self, file):
        for i in range(self.num_north):
            for j in range(self.num_east):
                # Write required bands
                for name in _LAYER_NAMES_REQUIRED:
                    file.write(struct.pack('h', self.layers[name].data[i,j].astype(_DATA_TYPE)))
                # Write crown fuel bands if present
                if self.crownPresent():
                    for name in _LAYER_NAMES_CROWN:
                        file.write(struct.pack('h', self.layers[name].data[i,j].astype(_DATA_TYPE)))
                # Write ground fuel bands if present
                if self.groundPresent():
                    for name in _LAYER_NAMES_GROUND:
                        file.write(struct.pack('h', self.layers[name].data[i,j].astype(_DATA_TYPE)))


    def writeLCP(self, filename):
        with open(filename, "wb") as file:
            self.__writeHeader(file)
            self.__writeBody(file)
    

    def write(self, prefix):
        self.writeLCP(prefix+".lcp")
        self.writeProjection(prefix+".prj")


    def writeNPY(self, prefix):
        # Write required layers
        # NOTE: Aspect refers to DOWN-SLOPE direction, measured CLOCKWISE from NORTH!
        
        # Perform unit conversions
        if self.layers['slope'].unit_opts == 0: # Data in degrees
            slope = -np.radians(self.layers['slope'].data)
        elif self.layers['slope'].unit_opts == 1: # Data in percent
            slope = -np.arctan(self.layers['slope'].data / 100)
        
        if self.layers['aspect'].unit_opts == 0: # Data in GRASS categories (1-25, 15 deg increments, 25=flat)
            aspect = np.radians(self.layers['aspect'].data * 15)
            # Handle flat case
            aspect[self.layers['aspect'].data == 25] = 0
            slope[self.layers['aspect'].data == 25] = 0
        elif self.layers['aspect'].unit_opts == 1: # Data in GRASS degress (CCW from EAST)
            aspect = np.radians(self.layers['aspect'].data)
        elif self.layers['aspect'].unit_opts == 2: # Data in azimuth degrees (CW from NORTH)
            aspect = -np.radians(self.layers['aspect'].data) + np.pi/2
        
        slope_east  = np.tan(slope) * np.cos(aspect)
        slope_north = np.tan(slope) * np.sin(aspect)
        np.save(prefix + "_slope_east", slope_east)
        np.save(prefix + "_slope_north", slope_north)
        np.save(prefix + "_elevation", self.layers['elevation'].data)
        np.save(prefix + "_fuel", self.layers['fuel'].data)
        np.save(prefix + "_cover", self.layers['cover'].data)
        # Write crown fuel layers if present
        if self.crownPresent():
            for name in _LAYER_NAMES_CROWN:
                np.save(prefix + "_" + name, self.layers[name].data)
        # Write ground fuel layers if present
        if self.groundPresent():
            for name in _LAYER_NAMES_GROUND:
                np.save(prefix + "_" + name, self.layers[name].data)


def main():
    pass


if __name__ == "__main__":
    main()
