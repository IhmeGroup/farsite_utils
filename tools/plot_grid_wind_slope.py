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
    parser.add_argument("-n", "--num-contours", type=int,
        dest='num_contours',
        required=False, default=5,
        help="Number of contours to plot")
    parser.add_argument('-o', '--out-file', type=str,
        dest='out_file',
        required=False, default="./result_grid",
        help="Output image filename (no extension)")
    
    args = parser.parse_args()

    case_ids = [int(id) for id in os.listdir(args.case_dir)]

    n_cases = len(case_ids)
    N = int(np.cbrt(n_cases))

    for i_slope in np.arange(N):
        print("Building figure {0}/{1}...".format(i_slope + 1, N+1))
        fig, axs = plt.subplots(
            nrows=N,
            ncols=N,
            figsize=(2.2*N, 2.25*N))

        for i_wind_speed in np.arange(N):
            for i_wind_direction in np.arange(N):

                case_id = (
                    N**0 * i_wind_speed +
                    N**1 * i_wind_direction +
                    N**2 * i_slope)
                
                print("Plotting case {0:03d}...".format(case_id))
                
                i = i_wind_direction
                j = i_wind_speed

                data = case.Case(os.path.join(args.case_dir, "{0:03d}".format(case_id), "job_farsite.slurm"))
                data.readOutput()

                slope = data.lcp.layers['slope'].data[0,0]

                x = np.arange(data.lcp.shape[0]) * data.lcp.res_x
                y = np.arange(data.lcp.shape[1]) * data.lcp.res_y
                X,Y = np.meshgrid(x,y)
                Y = np.flip(Y, axis=0)

                if (i == 0) and (j == 0):
                    fire_duration = np.amax(data.arrival_time.data)
                    contour_times = np.linspace(0.05*fire_duration, 0.95*fire_duration, args.num_contours)

                cmap = 'viridis'
                axs[i,j].imshow(data.arrival_time.data, extent=(0,x[-1],0,y[-1]), clim=(0, fire_duration), cmap=cmap)
                # axs[i,j].imshow(data.lcp.layers['elevation'].data, extent=(0,x[-1],0,y[-1]), cmap=cmap)
                # divider = make_axes_locatable(axs[i,j])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # m = plt.cm.ScalarMappable(cmap=cmap)
                # m.set_array(data.arrival_time.data)
                # cbar = plt.colorbar(m, cax=cax)
                # cbar.minorticks_on()
                if args.num_contours is not None:
                    axs[i,j].contour(X, Y, data.arrival_time.data, contour_times, colors='k')
                # if (i == 0):
                #     axs[i,j].set_title("Arrival time (min)")
                # axs[i,j].set_xticks(TICKS)
                # axs[i,j].set_yticks(TICKS)
                axs[i,j].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False)
        #         if (i == n_cases-1):
        #             axs[i,0].set_xlabel("4000m")
        #             axs[i,0].set_ylabel("4000m")
        #         if (i == n_cases-1):
        #             axs[i,0].tick_params(
        #                 labelbottom=True,
        #                 labelleft=True)
        #             axs[i,0].set_xticklabels(TICKLABELS)
        #             axs[i,0].set_yticklabels(TICKLABELS)
        # #             plt.setp(axs[i,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # #             plt.setp(axs[i,0].get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        #             axs[i,0].set_xlabel("$x$ (km)")
        #             axs[i,0].set_ylabel("$y$ (km)")

                windVec = np.array([
                    -data.weather.data['wind_speed'][0] * np.cos(
                        -np.radians(data.weather.data['wind_direction'][0]) + np.pi/2),
                    -data.weather.data['wind_speed'][0] * np.sin(
                        -np.radians(data.weather.data['wind_direction'][0]) + np.pi/2)])
                windMag = np.linalg.norm(windVec)
                unitWindVec = windVec / windMag
                northVec = np.array([0, 1])

                origin = np.array([x[-1]/8, y[-1]/8])
                N_color = 'w'
                speed_text = "{:0.1f} km/h".format(windMag)
                speed_offset = [700, 0]

        #         lw_arrow = 50.0
                lw_arrow = 30.0
        #         lw_arc = 2.25
                lw_arc = 2.0
        #         scale = 1000
                scale = 500
                axs[i,j].arrow(origin[0], origin[1], northVec[0]*scale, northVec[1]*scale, color='w', capstyle='round', width=lw_arrow, zorder=1001)
                axs[i,j].arrow(origin[0], origin[1], unitWindVec[0]*scale, unitWindVec[1]*scale, color='r', capstyle='round', width=lw_arrow, zorder=1002)
                theta2 = np.degrees(np.arctan2(windVec[1], windVec[0]))
                if theta2 < 0:
                    theta2 = theta2 + 360.0
                axs[i,j].add_patch(patches.Arc(origin, 0.5*scale, 0.5*scale, angle=0.0,
                    theta1=90.0, theta2=theta2, color='r', linewidth=lw_arc, zorder=1000))
                axs[i,j].text(origin[0], origin[1] + (northVec[1] + 0.2)*scale,
                    "N", color=N_color, ha='center', va='bottom', zorder=1003)
                # axs[i,j].text(
                #     origin[0] + unitWindVec[0]*scale*1.7 + speed_offset[0],
                #     origin[1] + unitWindVec[1]*scale*1.7 + speed_offset[1],
                #     speed_text,
                #     color='w', ha='center', va='center', zorder=1004)
                
                if i == 0:
                    axs[i,j].set_title("Wind Speed: {0:.1f} kmh".format(windMag))
                
                # if (i == 0) and (j == 3):
                #     import code; code.interact(local=locals())
        
        plt.suptitle("Slope: {0:.1f}".format(slope) + r"$^{\circ}$")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0)
        plt.savefig(
            args.out_file + "_{0}".format(i_slope),
            bbox_inches='tight',
            dpi=300)

if __name__=='__main__':
    main()
