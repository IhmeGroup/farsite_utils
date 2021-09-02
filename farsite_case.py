import landscape as ls
import raws
import generate as gen

class FarsiteCase:
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.lcp = ls.Landscape(prefix)
        self.wind = raws.RAWS(prefix + ".raws")
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def writeDir(self):
        """Write all files to directory with proper structure"""
        raise NotImplementedError