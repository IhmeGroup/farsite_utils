from farsite_utils import case
import os
import geopandas as gpd


def main():
    filepath = '/home/alvin/examples/Tubbs_Sims/'
    testCase = 'tubbs0' #change to plot different cases
    extension = '_Perimeters.shp'

    burnMap = case.Case()
    burnMap.perimeters = gpd.read_file(filepath + testCase + '/output/' + testCase + extension)
    burnMap.__convertAndMergePerimeters()
    burnMap.lcp.read(filepath + testCase + '/tubbsLCP')

    burnMap.computeBurnMaps()

    burnMap.renderOutput('testAnimate')





if __name__=='__main__':
    main()