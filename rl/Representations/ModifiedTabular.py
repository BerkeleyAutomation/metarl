"""Tabular representation"""

from rlpy.Representations import Tabular
import numpy as np
import os
import joblib

VALUES_FILE = "weights.p" # count might not be needed

class ModifiedTabular(Tabular):
    """
    Tabular representation that assigns a binary feature function f_{d}() 
    to each possible discrete state *d* in the domain. (For bounded continuous
    dimensions of s, discretize.)
    f_{d}(s) = 1 when d=s, 0 elsewhere.  (ie, the vector of feature functions
    evaluated at *s* will have all zero elements except one).
    NOTE that this representation does not support unbounded dimensions

    """

    def __init__(self, domain, discretization=20):
        super(ModifiedTabular, self).__init__(domain, discretization)

    def dump_to_directory(self, path):
        rep_values = {"weight_vec": self.weight_vec}

        joblib.dump(rep_values, 
                    os.path.join(path, VALUES_FILE))

    def load_from_directory(self, path):
        values_dir = joblib.load(os.path.join(path, VALUES_FILE))
        for attr, value in values_dir.items():
            self.__dict__[attr] = value
            print "Loaded {}".format(attr)