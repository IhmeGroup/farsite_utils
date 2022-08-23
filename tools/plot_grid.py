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

        data = case.Case(os.path.join(args.case_dir, case_id, "job.slurm"))
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
        if (i == n_cases-1):
            axs[i,0].set_xlabel("x (m)")
            axs[i,0].set_ylabel("y (m)")
        else:
            axs[i,0].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)

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
        axs[i,1].tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
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
        axs[i,2].tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        
        # import code; code.interact(local=locals())
        # print("Case: " + case_id + " - speed: {0:.2f} - direction: {1:.2f}".format(data.weather.data['wind_speed'][0], data.weather.data['wind_direction'][0]))
        # print("Case: " + case_id + " - fuel: {0:.2f}".format(data.lcp.layers['fuel'].data[0,0]))

        windVec = np.array([
            -data.weather.data['wind_speed'][0] * np.cos(
                -np.radians(data.weather.data['wind_direction'][0]) + np.pi/2),
            -data.weather.data['wind_speed'][0] * np.sin(
                -np.radians(data.weather.data['wind_direction'][0]) + np.pi/2)])
        windMag = np.linalg.norm(windVec)
        unitWindVec = windVec / windMag
        northVec = np.array([0, 1])
        origin = np.array([x[-1]/2, y[-1]/2])
        lw_arrow = 50.0
        lw_arc = 2.25
        scale = 1000
        axs[i,2].arrow(
            origin[0], origin[1],
            northVec[0]*scale, northVec[1]*scale,
            color='w', capstyle='round', width=lw_arrow, zorder=1001)
        axs[i,2].arrow(
            origin[0], origin[1],
            unitWindVec[0]*scale, unitWindVec[1]*scale,
            color='r', capstyle='round', width=lw_arrow, zorder=1002)
        theta2 = np.degrees(np.arctan2(windVec[1], windVec[0]))
        if theta2 < 0:
            theta2 = theta2 + 360.0
        axs[i,2].add_patch(patches.Arc(origin, 0.5*scale, 0.5*scale, angle=0.0,
            theta1=90.0, theta2=theta2, color='r', linewidth=lw_arc, zorder=1000))

        if (i == 0):
            N_color = 'k'
            speed_text = "{:0.1f} km/h".format(windMag)
            speed_offset = [500, 0]
        elif (i == 1):
            N_color = 'w'
            speed_text = "{:0.1f}\nkm/h".format(windMag)
            speed_offset = [100, 100]
        else:
            N_color = 'w'
            speed_text = "{:0.1f} km/h".format(windMag)
            speed_offset = [-300, 0]

        axs[i,2].text(origin[0], origin[1] + (northVec[1] + 0.2)*scale,
            "N", color=N_color, ha='center', va='bottom', zorder=1003)
        axs[i,2].text(
            origin[0] + unitWindVec[0]*scale*1.7 + speed_offset[0],
            origin[1] + unitWindVec[1]*scale*1.7 + speed_offset[1],
            speed_text,
            color='w', ha='center', va='center', zorder=1004)
    
    # axs[0,1].set_ylabel("\\emph{single fuel}")
    # axs[1,1].set_ylabel("\\emph{multiple fuel}")
    # axs[2,1].set_ylabel("\\emph{California}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(
        args.out_file,
        bbox_inches='tight',
        dpi=300)

if __name__=='__main__':
    main()
