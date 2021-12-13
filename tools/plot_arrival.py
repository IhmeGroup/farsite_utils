import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_metadata(file):
    metadata = dict()
    with open(file,'r') as f:
        for i in range(6):
            line = f.readline()
            words = line.split()
            if words[0] in ["NROWS", "NCOLS"]:
                metadata[words[0]] = int(words[1])
            else:
                metadata[words[0]] = float(words[1])
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Parse and plot arrival time.')
    parser.add_argument('-d', '--data-dir', type=str,
        dest='data_dir',
        required=False, default=os.getcwd(),
        help="Output data directory")
    parser.add_argument("-n", "--num-contours", type=int,
        dest='num_contours',
        required=False, default=None,
        help="Number of contours to plot")
    parser.add_argument('-o', '--out-file', type=str,
        dest='out_file',
        required=False, default="./arrival_time",
        help="Output image filename (no extension)")
    
    args = parser.parse_args()

    for file in os.listdir(args.data_dir):
        if file[-15:] == "ArrivalTime.asc":
            arrival_time_file = file
            break
    
    # Parse NODATA value
    metadata = parse_metadata(os.path.join(args.data_dir, arrival_time_file))

    # Read data
    arrival_time = np.loadtxt(
        os.path.join(args.data_dir, arrival_time_file),
        skiprows=6)
    arrival_time = np.flip(arrival_time, axis=0)
    # arrival_time[np.isclose(arrival_time, metadata['NODATA_VALUE'])] = 0.0
    arrival_time_masked = np.ma.masked_where(
        np.isclose(arrival_time, metadata['NODATA_VALUE']),
        arrival_time)

    #
    x = np.arange(metadata['NROWS']) * metadata['CELLSIZE']
    y = np.arange(metadata['NCOLS']) * metadata['CELLSIZE']
    X,Y = np.meshgrid(x,y)
    
    fire_duration = np.amax(arrival_time)
    contour_times = np.linspace(0, fire_duration, args.num_contours)
    
    plt.imshow(
        np.zeros_like(arrival_time),
        cmap='gray',
        extent=(0,x[-1],0,y[-1]))
    plt.imshow(arrival_time_masked, origin='lower', extent=(0,x[-1],0,y[-1]))
    plt.colorbar(label="Arrival time (min)")
    if args.num_contours is not None:
        plt.contour(X, Y, arrival_time_masked, contour_times, colors='k')
    plt.xlabel("y (m)")
    plt.ylabel("x (m)")
    plt.savefig(
        args.out_file,
        bbox_inches='tight',
        dpi=300)
    plt.show()

if __name__=='__main__':
    main()