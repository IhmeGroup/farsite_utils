"""This module contains utilities for managing ensembles of FARSITE cases."""

from enum import Enum
import os
import copy
import multiprocessing
import numpy as np
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
        finalBurnFraction = np.zeros((self.size))
        for i, case in enumerate(self.cases):
            case.readOutput()
            finalBurnFraction[i] = case.finalBurnFraction()
        
        finalBurnFraction_nonan = finalBurnFraction[~np.isnan(finalBurnFraction)]

        mu = finalBurnFraction_nonan.mean()
        median = np.median(finalBurnFraction_nonan)
        sigma = finalBurnFraction_nonan.std()

        textstr = '\n'.join((
            "$\mu = {0:.2f}$".format(mu),
            "$\mathrm{median} = " + "{0:.2f}$".format(median),
            "$\sigma = {0:.2f}$".format(sigma)))

        fig, ax = plt.subplots()
        ax.hist(finalBurnFraction_nonan, bins=np.linspace(0, 1, 11))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=_SMALL_SIZE,
                verticalalignment='top', bbox=props)
        ax.set_xlabel("Final Burn Coverage")
        ax.set_ylabel("Occurences")

        import code; code.interact(local=locals())
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.root_dir, self.out_dir_local, "stats.png"),
            bbox_inches='tight',
            dpi=300)


def main():
    pass


if __name__ == "__main__":
    main()
