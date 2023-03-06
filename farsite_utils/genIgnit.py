from distutils.command.install_egg_info import to_filename
from farsite_utils import case
from farsite_utils import generate
import geopandas as gpd
import numpy as np

flameCase = case.Case()
flameCase.lcp.read('/home/alvin/Datasets2/LCP_US_140FBFM40/LCP_US_140FBFM40')
print(flameCase.lcp.utm_east)
print(flameCase.lcp.utm_west)

flameCase.ignition = gpd.read_file('/home/alvin/Datasets2/LCP_US_140FBFM40/metadata.shp')
accessible_fraction = 0.5
print((flameCase.lcp.utm_east + flameCase.lcp.utm_west)/2, (flameCase.lcp.utm_north + flameCase.lcp.utm_south)/2 )
width = flameCase.lcp.utm_east - flameCase.lcp.utm_west
height = flameCase.lcp.utm_north - flameCase.lcp.utm_south
flameCase.ignition.geometry[0] = generate.regularPolygon(
            sides       = 8,
            radius      = .01*(flameCase.lcp.utm_east - flameCase.lcp.utm_west),
            rotation    = 0,
            #translation = ((flameCase.lcp.utm_east + flameCase.lcp.utm_west)/2, (flameCase.lcp.utm_north + flameCase.lcp.utm_south)/2 ))
            translation = (
                np.random.uniform(
                    flameCase.lcp.utm_west + (accessible_fraction/2)*width, 
                    flameCase.lcp.utm_east - (accessible_fraction/2)*width),
                np.random.uniform(
                    flameCase.lcp.utm_south + (accessible_fraction/2)*height,
                    flameCase.lcp.utm_north - (accessible_fraction/2)*height)))


print(flameCase.ignition)
flameCase.ignition.to_file('~/Datasets2/TubbsFarsite.shp')