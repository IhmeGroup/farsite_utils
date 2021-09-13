"""This module contains utilities for managing ensembles of FARSITE cases."""

import os
import copy
import multiprocessing
import numpy as np

from . import case

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
    

    @property
    def size(self):
        """The number of cases in the ensemble."""
        return len(self.cases)
    

    @property
    def name(self):
        """The name of this ensemble."""
        return self._name
    

    @name.setter
    def name(self, value):
        """Set the name for this ensemble and all of its cases."""
        self._name = value
        for i, case in enumerate(self.cases):
            case.name = value + "_{0:0{1}d}".format(i, int(np.ceil(np.log10(self.size-1))))


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
                "{0:0{1}d}".format(i, int(np.ceil(np.log10(self.size-1)))))
    

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
    

    def write(self):
        """Write all cases in ensemble."""
        for case in self.cases:
            if not os.path.isdir(case.root_dir):
                os.makedirs(case.root_dir, exist_ok=True)
            case.write()
    

    def run(self):
        """Run all cases in ensemble."""
        for case in self.cases:
            case.run()
    

    def postProcessCase(self, case):
        """Postprocess a single case and return success status."""
        if not case.isDone():
            return False
        case.readOutput()
        case.renderOutput(os.path.join(case.root_dir, case.name))
        case.computeBurnMaps()
        case.exportData(os.path.join(self.root_dir, self.out_dir_local, case.name))
        return True
    

    def postProcess(self, n_processes=multiprocessing.cpu_count()):
        """Postprocess all cases in ensemble."""
        pool = multiprocessing.Pool(n_processes)
        self.exported = pool.map(self.postProcessCase, self.cases)

        print("Failed to postprocess cases:")
        for i in range(self.size):
            if not self.exported[i]:
                print(i, end=" ")


def main():
    pass


if __name__ == "__main__":
    main()
