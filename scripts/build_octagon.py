import geopandas as gpd
from shapely.ops import unary_union

import landscape as ls
import generate as gen

ls1 = ls.Landscape("./data/landscapes/ref_128")
gdf = gpd.read_file("./data/shapes/flatland_centerIgnit.shp")

center_x, center_y = ls1.center
radius = float(gdf.geometry.bounds['maxy'] - gdf.geometry.bounds['miny'])

geom = gen.regularPolygon(8, radius, 0, ls1.center)

# octagon1 = gen.regularPolygon(
#     8,
#     radius,
#     0,
#     (center_x, center_y - 1.5*radius))

# octagon2 = gen.regularPolygon(
#     8,
#     radius,
#     0,
#     (center_x, center_y + 1.5*radius))

# geom = unary_union([octagon1, octagon2])

gdf.geometry[0] = geom
gdf.to_file("./data/shapes/octagon_128.shp")
