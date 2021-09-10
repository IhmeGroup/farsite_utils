import landscape
import numpy as np

shape = (128, 128)
ltest = landscape.Landscape(shape=shape)
ltest.description = "Reference flat 128x128 landscape"
ltest.readProjection("./data/landscapes/flatland.prj")
ltest.lo_east = 0.0
ltest.hi_east = 3840.0
ltest.lo_north = 0.0
ltest.hi_north = 3840.0
ltest.utm_west = 20000.0
ltest.utm_east = 23840.0
ltest.utm_south = 20000.0
ltest.utm_north = 23840.0
ltest.units_grid = 0
ltest.res_x = 30.0
ltest.res_y = 30.0
ltest.layers['elevation'].file = ""
ltest.layers['elevation'].unit_opts = np.int16(0) # Meters
ltest.layers['elevation'].data = np.zeros(shape, dtype=np.int16) + 100
ltest.layers['slope'].file = ""
ltest.layers['slope'].unit_opts = np.int16(0) # Degrees
ltest.layers['slope'].data = np.zeros(shape, dtype=np.int16) + 0
ltest.layers['aspect'].file = ""
ltest.layers['aspect'].unit_opts = np.int16(0) # Degrees
ltest.layers['aspect'].data = np.zeros(shape, dtype=np.int16) + 0
ltest.layers['fuel'].file = ""
ltest.layers['fuel'].unit_opts = np.int16(0) # No custom and no file
ltest.layers['fuel'].data = np.zeros(shape, dtype=np.int16) + 1
ltest.layers['cover'].file = ""
ltest.layers['cover'].unit_opts = np.int16(0) # Percent
ltest.layers['cover'].data = np.zeros(shape, dtype=np.int16) + 50
ltest.crown_fuels = 21 # on
ltest.layers['height'].file = ""
ltest.layers['height'].unit_opts = np.int16(3) # Meters x 10
ltest.layers['height'].data = np.zeros(shape, dtype=np.int16) + 170
ltest.layers['base'].file = ""
ltest.layers['base'].unit_opts = np.int16(3) # Meters x 10
ltest.layers['base'].data = np.zeros(shape, dtype=np.int16) + 10
ltest.layers['density'].file = ""
ltest.layers['density'].unit_opts = np.int16(3) # kg / m^3
ltest.layers['density'].data = np.zeros(shape, dtype=np.int16) + 20
ltest.ground_fuels = 20 # off
ltest.layers['duff'].file = ""
ltest.layers['duff'].data = np.zeros(shape, dtype=np.int16)
ltest.layers['woody'].file = ""
ltest.layers['woody'].data = np.zeros(shape, dtype=np.int16)
ltest.write("./data/landscapes/ref_128")