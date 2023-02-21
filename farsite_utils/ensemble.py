"""This module contains utilities for managing ensembles of FARSITE cases."""

from enum import Enum
import os
import copy
import functools
import multiprocessing
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager

from . import case

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

_XSMALL_SIZE = 12
_SMALL_SIZE = 14
_MEDIUM_SIZE = 16
_BIGGER_SIZE = 18

plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=_SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=_XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title


class CaseStatus(Enum):
    IGNITION_FAILED = 1
    NOT_DONE_YET = 2
    DONE = 3


class Ensemble:
    """This class represents an ensemble of FARSITE cases."""

    def __init__(self, name=None, root_dir=None, n_cases=0, prototype=None):
        self.cases = np.array([copy.deepcopy(prototype) for i in range(n_cases)])
        self.cases_dir_local = "./cases"
        self.out_dir_local = "./export"
        if name:
            self.name = name
        if root_dir:
            self.root_dir = root_dir
        self.exported = [False] * self.size
        self.post_windninja = [False] * self.size
        self.verbose = False
    

    @property
    def size(self):
        """The number of cases in the ensemble."""
        return len(self.cases)
    

    def caseID(self, i):
        """Return formatted case ID string"""
        return "{0:0{1}d}".format(i, int(np.ceil(np.log10(self.size-1))))
    

    @property
    def name(self):
        """The name of this ensemble."""
        return self._name
    

    @name.setter
    def name(self, value):
        """Set the name for this ensemble and all of its cases."""
        self._name = value
        for i, case in enumerate(self.cases):
            case.name = value + "_" + self.caseID(i)


    @property
    def root_dir(self):
        """The root directory for this ensemble."""
        return self._root_dir
    

    @root_dir.setter
    def root_dir(self, value):
        """Set the root dir for this ensemble."""
        self._root_dir = value
        for i, case in enumerate(self.cases):
            case.root_dir = os.path.join(
                value,
                self.cases_dir_local,
                self.caseID(i))
    

    @property
    def cases(self):
        """The list of Cases."""
        return self._cases
    

    @cases.setter
    def cases(self, value):
        """Set the entire list of Cases."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Ensemble.cases must be a 1D np.array of Cases.")
        if value.ndim != 1:
            raise TypeError("Ensemble.cases must be a 1D np.array of Cases.")
        for item in value:
            if not isinstance(item, case.Case):
                raise TypeError("Ensemble.cases must be a 1D np.array of Cases.")
        self._cases = value
    

    @property
    def verbose(self):
        return self._verbose
    

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            raise TypeError("Ensemble.verbose must be a bool.")
        self._verbose = value
        for case in self.cases:
            case.verbose = True
    

    def writeCase(self, case):
        """Write a single case."""
        if not os.path.isdir(case.root_dir):
            os.makedirs(case.root_dir, exist_ok=True)
        case.write()
    

    def write(self, case_ids=None, n_processes=multiprocessing.cpu_count()):
        """Write all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        pool = multiprocessing.Pool(n_processes)
        pool.map(self.writeCase, self.cases[case_ids])
    

    def run(self, case_ids=None):
        """Run all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        for case in self.cases[case_ids]:
            case.run()
    

    def runWindNinja(self, case_ids=None):
        """Run WindNinja on all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        for case in self.cases[case_ids]:
            case.runWindNinja()
    

    def clearWindNinja(self, case_ids=None):
        """Clear WindNinja data for all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        for case in self.cases[case_ids]:
            case.clearWindNinja()
    

    def postProcessCase(self, case, render=False):
        """Postprocess a single case and return success status."""
        if case.ignitionFailed():
            return CaseStatus.IGNITION_FAILED
        if not case.isDone():
            return CaseStatus.NOT_DONE_YET
        case.readOutput()
        if case.windninja_active:
            case.readOutputWindNinjaPost()
        if render:
            case.renderOutput(os.path.join(case.root_dir, case.name))
        case.computeBurnMaps()
        case.exportData(os.path.join(self.root_dir, self.out_dir_local, case.name))
        return CaseStatus.DONE
    

    def postProcess(self, case_ids=None, render=False, n_processes=multiprocessing.cpu_count(), 
                    attempts=1, pause_time=5):
        """Postprocess all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        for i in range(attempts):
            print("> postProcess: Attempt {0}".format(i))
            # Collect cases which have not yet been exported
            indices_to_export = []
            cases_to_export = []
            for j, case in enumerate(self.cases[case_ids]):
                if not self.exported[j]:
                    indices_to_export.append(case_ids[j])
                    cases_to_export.append(case)
            
            # Postprocess remaining cases in parallel
            pool = multiprocessing.Pool(n_processes)
            results = pool.map(
                functools.partial(
                    self.postProcessCase,
                    render=render),
                cases_to_export)

            # Determine which cases have been successful
            cases_ignition_failed = []
            cases_not_done_yet = []
            for j, case_result in enumerate(results):
                self.exported[indices_to_export[j]] = (case_result == CaseStatus.DONE)
                if case_result == CaseStatus.IGNITION_FAILED:
                    cases_ignition_failed.append(self.caseID(indices_to_export[j]))
                elif case_result == CaseStatus.NOT_DONE_YET:
                    cases_not_done_yet.append(self.caseID(indices_to_export[j]))
            
            # If no remaining cases, break. Otherwise report failure and try again
            if (not cases_ignition_failed) and (not cases_not_done_yet):
                break
            else:
                print("Failed to ignite:", *cases_ignition_failed)
                print("Failed to postprocess:", *cases_not_done_yet)
                time.sleep(pause_time)
        
        return cases_ignition_failed + cases_not_done_yet
    

    def postProcessWindNinjaCase(self, case):
        """Postprocess WindNinja for a single case and return success status."""
        if not case.isDoneWindNinja():
            return CaseStatus.NOT_DONE_YET
        case.readOutputWindNinja()
        case.expandWindNinjaData()
        case.atm.write(os.path.join(case.root_dir, case.name + ".atm"))
        case.writeInput(os.path.join(case.root_dir, case.name + ".input"))
        return CaseStatus.DONE
    

    def postProcessWindNinja(self, case_ids=None, n_processes=multiprocessing.cpu_count(), 
                    attempts=1, pause_time=5):
        """Postprocess WindNinja for all cases in ensemble."""
        if case_ids is None:
            case_ids = np.arange(self.size)
        for i in range(attempts):
            print("> postProcessWindNinja: Attempt {0}".format(i))
            # Collect cases which have not yet been exported
            indices_to_export = []
            cases_to_export = []
            for j, case in enumerate(self.cases[case_ids]):
                if not self.post_windninja[j]:
                    indices_to_export.append(case_ids[j])
                    cases_to_export.append(case)
            
            # Postprocess remaining cases in parallel
            pool = multiprocessing.Pool(n_processes)
            results = pool.map(
                functools.partial(
                    self.postProcessWindNinjaCase),
                cases_to_export)

            # Determine which cases have been successful
            cases_not_done_yet = []
            for j, case_result in enumerate(results):
                self.post_windninja[indices_to_export[j]] = (case_result == CaseStatus.DONE)
                if case_result == CaseStatus.NOT_DONE_YET:
                    cases_not_done_yet.append(self.caseID(indices_to_export[j]))
            
            # If no remaining cases, break. Otherwise report failure and try again
            if not cases_not_done_yet:
                break
            else:
                print("Failed to postprocess - windninja:", *cases_not_done_yet)
                time.sleep(pause_time)
        
        return cases_not_done_yet
    

    def __computeAndPlotHistogram(self, data, name, title, bins=None):
        n = len(data)

        # Compute histograms
        hist = np.zeros((n, len(bins)-1))
        edges = np.zeros((n+1, len(bins)))
        time = np.zeros((n+1, len(bins)))
        for i in range(n):
            hist[i,:], edges[i,:] = np.histogram(data.iloc[i], bins=bins)
            time[i,:] = i
        time[n, :] = n
        edges[n, :] = edges[n-1, :]

        # Compute aggregate statistics
        mu = np.nanmean(data.to_numpy(), axis=1)
        median = np.nanmedian(data.to_numpy(), axis=1)
        sigma = np.nanstd(data.to_numpy(), axis=1)
        bound_u = mu + 2*sigma
        bound_l = mu - 2*sigma

        data['mu'] = mu
        data['median'] = median
        data['sigma'] = sigma

        # Write stats to file
        data.to_csv(os.path.join(self.root_dir, self.out_dir_local, "stats_" + name + ".csv"))

        # Plot stats
        time_vec = np.arange(0, n)
        lw = 3
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        im = axs[0].pcolor(edges, time, hist)
        plt.colorbar(im, ax=axs[0], label="Occurrences")
        axs[0].set_xlabel(title)
        axs[0].set_ylabel("Step")
        axs[0].plot(mu, time_vec, color='magenta', linewidth=lw, label="$\mu$")
        axs[0].plot(median, time_vec, color='r', linewidth=lw, label="median")
        axs[0].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
        axs[0].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
        axs[0].legend(loc='lower right')

        hist_norm = hist / hist.max(axis=1, keepdims=True)
        im = axs[1].pcolor(edges, time, hist_norm)
        plt.colorbar(im, ax=axs[1], label="Occurrences / Max Occurrences")
        axs[1].set_xlabel(title)
        axs[1].set_ylabel("Step")
        axs[1].plot(mu, time_vec, color='magenta', linewidth=lw, label="$\mu$")
        axs[1].plot(median, time_vec, color='r', linewidth=lw, label="median")
        axs[1].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
        axs[1].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
        axs[1].legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.root_dir, self.out_dir_local, "stats_" + name + ".png"),
            bbox_inches='tight',
            dpi=300)
    

    def computeStatistics(self):

        # Compute burn fraction over time for each case
        burn_fraction = pd.DataFrame()
        for i, case in enumerate(self.cases):
            print("Computing statistics: case " + self.caseID(i))

            try:
                case.readOutput()
            except FileNotFoundError:
                burn_fraction[self.caseID(i)] = np.nan
                continue
            
            burn_fraction_i = case.burnFraction()
            if len(burn_fraction_i) > len(burn_fraction):
                burn_fraction = burn_fraction.reindex(list(range(0, len(burn_fraction_i)))).reset_index(drop=True)
            else:
                burn_fraction[self.caseID(i)] = np.nan

            burn_fraction.loc[0:len(burn_fraction_i)-1, self.caseID(i)] = burn_fraction_i

        burn_radius = np.sqrt(burn_fraction * self.cases[0].lcp.area / np.pi)
        front_speed = pd.DataFrame(0, index=burn_radius.index, columns=burn_radius.columns)
        front_speed[:] = np.gradient(burn_radius, self.cases[0].timestep, axis=0)

        n_bins = 21
        self.__computeAndPlotHistogram(
            burn_fraction,
            "burn_fraction",
            "Burned Area Fraction",
            bins=np.linspace(0, 1, n_bins))
        self.__computeAndPlotHistogram(
            burn_radius,
            "burn_radius",
            "Burned Area Equivalent Circle Radius",
            bins=np.linspace(0, burn_radius.dropna(axis=1).to_numpy().max()))
        self.__computeAndPlotHistogram(
            front_speed,
            "front_speed",
            "Effective Front Speed",
            bins=np.linspace(0, front_speed.dropna(axis=1).to_numpy().max()))
    

    def exportProfilingData(self):

        # Get profiling data for each case
        step_wtimes = []
        for i, case in enumerate(self.cases):
            print("Gathering profiling data: case " + self.caseID(i))

            try:
                case.readProfilingData()
            except FileNotFoundError:
                continue

            step_wtimes += case.step_wtimes
        
        profiling_data = pd.DataFrame()
        profiling_data['wtime'] = step_wtimes
        profiling_data.to_csv(os.path.join(self.root_dir, self.out_dir_local, "profiling_data.csv"))

        mu = np.mean(profiling_data['wtime'])
        median = np.median(profiling_data['wtime'])
        sigma = np.std(profiling_data['wtime'])

        fig, ax = plt.subplots()
        ax.hist(profiling_data['wtime']*1000.0, bins=50)
        ax.text(
            0.9, 0.9,
            "$\mu = {0:.1f}$\nmedian$ = {1:.1f}$\n$\sigma = {2:.1f}$".format(mu*1000, median*1000, sigma*1000),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes)
        ax.set_xlabel("Step Wall Time (ms)")
        ax.set_ylabel("Occurences")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.root_dir, self.out_dir_local, "profiling_data.png"),
            bbox_inches='tight',
            dpi=300)


def main():
    pass


if __name__ == "__main__":
    main()
