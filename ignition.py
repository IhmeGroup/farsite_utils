"""Class for managing ignition shapes."""
import geopandas as gpd


class Ignition:
    def __init__(self, filename=None):
        self.filename = filename
        self.gdf = gpd.GeoDataFrame

        if self.filename:
            self.read(self.filename)
    

    def read(self, filename):
        self.gdf = gpd.read_file(filename)
    

    def write(self, filename):
        self.gdf.to_file(filename)


def main():
    pass


if __name__ == "__main__":
    main()
