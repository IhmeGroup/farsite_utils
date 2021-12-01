"""This module contains utilities for managing ensembles of FARSITE cases."""

from enum import Enum
import os
import copy
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
        self.cases = [copy.deepcopy(prototype) for i in range(n_cases)]
        self.cases_dir_local = "./cases"
        self.out_dir_local = "./export"
        if name:
            self.name = name
        if root_dir:
            self.root_dir = root_dir
        self.exported = [False] * self.size
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
        if not isinstance(value, list):
            raise TypeError("Ensemble.cases must be a list of Cases.")
        for item in value:
            if not isinstance(item, case.Case):
                raise TypeError("Ensemble.cases must be a list of Cases.")
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
    

    def write(self, n_processes=multiprocessing.cpu_count()):
        """Write all cases in ensemble."""
        pool = multiprocessing.Pool(n_processes)
        pool.map(self.writeCase, self.cases)
    

    def run(self):
        """Run all cases in ensemble."""
        for case in self.cases:
            case.run()
    

    def postProcessCase(self, case):
        """Postprocess a single case and return success status."""
        if case.ignitionFailed():
            return CaseStatus.IGNITION_FAILED
        if not case.isDone():
            return CaseStatus.NOT_DONE_YET
        case.readOutput()
        case.renderOutput(os.path.join(case.root_dir, case.name))
        case.computeBurnMaps()
        case.exportData(os.path.join(self.root_dir, self.out_dir_local, case.name))
        return CaseStatus.DONE
    

    def postProcess(self, n_processes=multiprocessing.cpu_count(), attempts=1, pause_time=5):
        """Postprocess all cases in ensemble."""
        for i in range(attempts):
            print("Attempt {0}".format(i))
            # Collect cases which have not yet been exported
            indices_to_export = []
            cases_to_export = []
            for j, case in enumerate(self.cases):
                if not self.exported[j]:
                    indices_to_export.append(j)
                    cases_to_export.append(case)
            
            # Postprocess remaining cases in parallel
            pool = multiprocessing.Pool(n_processes)
            results = pool.map(self.postProcessCase, cases_to_export)

            # Determine which cases have been successful
            cases_ignition_failed = []
            cases_not_done_yet = []
            for j, case_result in enumerate(results):
                self.exported[indices_to_export[j]] = (case_result == CaseStatus.DONE)
                if case_result == CaseStatus.IGNITION_FAILED:
                    cases_ignition_failed.append(self.caseID(indices_to_export[j]))
                elif case_result == CaseStatus.NOT_DONE_YET:
                    cases_not_done_yet.append(self.caseID(indices_to_export[j]))
            
            # If all exported, break. Otherwise report failure and try again
            if all(self.exported):
                for i in range(self.size):
                    self.exported[i] = True
                break
            else:
                print("Failed to ignite:", *cases_ignition_failed)
                print("Failed to postprocess:", *cases_not_done_yet)
                time.sleep(pause_time)
    

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

            # if i > 100:
            #     break

        # Compute histograms
        bins = np.linspace(0, 1, 21)
        hist = np.zeros((len(burn_fraction), len(bins)-1))
        edges = np.zeros((len(burn_fraction)+1, len(bins)))
        time = np.zeros((len(burn_fraction)+1, len(bins)))
        for i in range(len(burn_fraction)):
            hist[i,:], edges[i,:] = np.histogram(burn_fraction.iloc[i], bins=bins)
            time[i,:] = i
        time[len(burn_fraction), :] = len(burn_fraction)
        edges[len(burn_fraction), :] = edges[len(burn_fraction)-1, :]
        
        # Compute aggregate statistics
        mu = np.nanmean(burn_fraction.to_numpy(), axis=1)
        median = np.nanmedian(burn_fraction.to_numpy(), axis=1)
        sigma = np.nanstd(burn_fraction.to_numpy(), axis=1)
        bound_u = mu + 2*sigma
        bound_l = mu - 2*sigma
        bound_l[bound_l < 0] = np.nan

        burn_fraction['mu'] = mu
        burn_fraction['median'] = median
        burn_fraction['sigma'] = sigma

        # Write stats to file
        burn_fraction.to_csv(os.path.join(self.root_dir, self.out_dir_local, "stats.csv"))

        # Plot stats
        time_vec = np.arange(0, len(burn_fraction))
        lw = 3
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        im = axs[0].pcolor(edges, time, hist)
        plt.colorbar(im, ax=axs[0], label="Occurences")
        axs[0].set_xlabel("Burned Area Fraction")
        axs[0].set_ylabel("Step")
        axs[0].plot(mu, time_vec, color='magenta', linewidth=lw, label="$\mu$")
        axs[0].plot(median, time_vec, color='r', linewidth=lw, label="median")
        axs[0].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
        axs[0].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
        axs[0].legend(loc='lower right')

        hist_norm = hist / hist.max(axis=1, keepdims=True)
        im = axs[1].pcolor(edges, time, hist_norm)
        plt.colorbar(im, ax=axs[1], label="Occurences / Max Occurences")
        axs[1].set_xlabel("Burned Area Fraction")
        axs[1].set_ylabel("Step")
        axs[1].plot(mu, time_vec, color='magenta', linewidth=lw, label="$\mu$")
        axs[1].plot(median, time_vec, color='r', linewidth=lw, label="median")
        axs[1].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
        axs[1].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
        axs[1].legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.root_dir, self.out_dir_local, "stats.png"),
            bbox_inches='tight',
            dpi=300)


def main():
    pass


if __name__ == "__main__":
    main()
