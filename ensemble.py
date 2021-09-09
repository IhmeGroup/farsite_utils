"""Class representing an ensemble of Farsite simulations"""

import case
import raws
import landscape


class Ensemble:
    def __init__(self):
        self.prototype = case.Case()
        self.cases = []
        self.gen_weather = lambda : raws.RAWS()
        self.gen_landscape = lambda : landscape.Landscape()
        raise NotImplementedError