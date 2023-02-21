import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from farsite_utils import case

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title

TICKS = [0, 1000, 2000, 3000]
TICKLABELS = ["0", "1", "2", "3"]

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
    parser.add_argument('-d', '--case-dir', type=str,
        dest='case_dir',
        required=True, default=os.getcwd(),
        help="Fire case directory")
    parser.add_argument('-c', '--case-ids', type=str,
        dest='case_ids', nargs='*',
        required=True, default=["000"],
        help="Fire case directory")
    parser.add_argument("-n", "--num-contours", type=int,
        dest='num_contours',
        required=False, default=5,
        help="Number of contours to plot")
    parser.add_argument('-o', '--out-file', type=str,
        dest='out_file',
        required=False, default="./result_grid",
        help="Output image filename (no extension)")
    
    args = parser.parse_args()

    # case_dirs_exp = []
    # for path in args.case_dirs:
    #     case_dirs_exp += glob.glob(path)

    n_cases = len(args.case_ids)

    cols = 3
    fig, axs = plt.subplots(
        nrows=n_cases,
        ncols=cols,
        figsize=(2.8*cols, 2.25*n_cases))

    for i, case_id in enumerate(args.case_ids):

        data = case.Case(os.path.join(args.case_dir, case_id, "job_farsite.slurm"))
        data.readOutputWindNinja()
        data.readOutput()

        x = np.arange(data.lcp.shape[0]) * data.lcp.res_x
        y = np.arange(data.lcp.shape[1]) * data.lcp.res_y
        X,Y = np.meshgrid(x,y)
        Y = np.flip(Y, axis=0)

        fire_duration = np.amax(data.arrival_time.data)
        contour_times = np.linspace(0.05*fire_duration, 0.95*fire_duration, args.num_contours)

        # data.verbose = True
        # data.computeBurnMaps()
        # import code; code.interact(local=locals())

        cmap = 'plasma'
        axs[i,0].imshow(data.arrival_time.data, extent=(0,x[-1],0,y[-1]), cmap=cmap)
        divider = make_axes_locatable(axs[i,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(data.arrival_time.data)
        cbar = plt.colorbar(m, cax=cax)
        cbar.minorticks_on()
        if args.num_contours is not None:
            axs[i,0].contour(X, Y, data.arrival_time.data, contour_times, colors='k')
        if (i == 0):
            axs[i,0].set_title("Arrival time (min)")
        axs[i,0].set_xticks(TICKS)
        axs[i,0].set_yticks(TICKS)
        axs[i,0].tick_params(
            axis='both',
            which='both',
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=False,
            labelleft=False)
#         if (i == n_cases-1):
#             axs[i,0].set_xlabel("4000m")
#             axs[i,0].set_ylabel("4000m")
        if (i == n_cases-1):
            axs[i,0].tick_params(
                labelbottom=True,
                labelleft=True)
            axs[i,0].set_xticklabels(TICKLABELS)
            axs[i,0].set_yticklabels(TICKLABELS)
#             plt.setp(axs[i,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#             plt.setp(axs[i,0].get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            axs[i,0].set_xlabel("$x$ (km)")
            axs[i,0].set_ylabel("$y$ (km)")

        cmap = 'viridis'
        im = axs[i,1].imshow(data.lcp.layers['fuel'].data, extent=(0,x[-1],0,y[-1]), cmap=cmap)
        im.set_clim(101, 205)
        divider = make_axes_locatable(axs[i,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_clim(101, 205)
        m.set_array(data.lcp.layers['fuel'].data)
        cbar = plt.colorbar(m, cax=cax)
        cbar.minorticks_on()
        if args.num_contours is not None:
            axs[i,1].contour(X, Y, data.arrival_time.data, contour_times, colors='k')
        if (i == 0):
            axs[i,1].set_title("Fuel")
        axs[i,1].set_xticks(TICKS)
        axs[i,1].set_yticks(TICKS)
        axs[i,1].tick_params(
            axis='both',
            which='both',
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=False,
            labelleft=False)

        cmap = 'gray'
        axs[i,2].imshow(data.lcp.layers['elevation'].data, extent=(0,x[-1],0,y[-1]), cmap=cmap)
        # axs[i,2].imshow(data.burn[100], origin='lower', extent=(0,x[-1],0,y[-1]), cmap=cmap)
        divider = make_axes_locatable(axs[i,2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(data.lcp.layers['elevation'].data)
        cbar = plt.colorbar(m, cax=cax)
        cbar.minorticks_on()
        if args.num_contours is not None:
            axs[i,2].contour(X, Y, data.arrival_time.data, contour_times, colors='k')
        if (i == 0):
            axs[i,2].set_title("Elevation (m)")
        axs[i,2].set_xticks(TICKS)
        axs[i,2].set_yticks(TICKS)
        axs[i,2].tick_params(
            axis='both',
            which='both',
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=False,
            labelleft=False)

        wind_n_plot = 10
        wind_nrows = data.atm.data['wind_speed_data'][0].nrows
        wind_ncols = data.atm.data['wind_speed_data'][0].ncols
        wind_interval_x = int(np.floor(wind_nrows / wind_n_plot))
        wind_interval_y = int(np.floor(wind_ncols / wind_n_plot))
        wind_cell_size = data.atm.data['wind_speed_data'][0].cell_size
        wind_x = np.arange(wind_nrows) * wind_cell_size
        wind_y = np.arange(wind_ncols) * wind_cell_size
        wind_X, wind_Y = np.meshgrid(wind_x, wind_y)
        wind_Y = np.flip(wind_Y, axis=0)
        wind_speed     = data.atm.data['wind_speed_data'    ][0].data
        wind_direction = data.atm.data['wind_direction_data'][0].data
        wind_east  = -wind_speed * np.cos(-np.radians(wind_direction) + np.pi/2)
        wind_north = -wind_speed * np.sin(-np.radians(wind_direction) + np.pi/2)

        axs[i,2].quiver(
            wind_X[::wind_interval_x, ::wind_interval_y],
            wind_Y[::wind_interval_x, ::wind_interval_y],
            wind_east[ ::wind_interval_x, ::wind_interval_y],
            wind_north[::wind_interval_x, ::wind_interval_y],
            color='r',
            pivot='middle',
            scale=300,
            scale_units='width',
            width=0.01,
            zorder=np.inf)
        axs[i,2].set_xlim(0, x[-1])
        axs[i,2].set_ylim(0, y[-1])
    
    # axs[0,0].set_ylabel("\\emph{single fuel}")
    # axs[1,0].set_ylabel("\\emph{multiple fuel}")
    # axs[2,0].set_ylabel("\\emph{California}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(
        args.out_file,
        bbox_inches='tight',
        dpi=300)

if __name__=='__main__':
    main()
