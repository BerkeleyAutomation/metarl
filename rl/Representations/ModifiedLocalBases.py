from rlpy.Representations import RandomLocalBases
import joblib, os
import matplotlib.pyplot as plt
VALUES_FILE = "weights.p"

class ModifiedRandomLocalBases(RandomLocalBases):

    initialized = False
    def __init__(self, domain, kernel, **kwargs):
        super(ModifiedRandomLocalBases, self).__init__(domain, kernel, **kwargs)

    def init_randomization(self):
        if self.initialized:
            print "ModifiedRandomLocalBases already initialized!!!!!"
            import time; time.sleep(1)
        else:
            super(ModifiedRandomLocalBases, self).init_randomization()
            self.initialized = True

    def dump_to_directory(self, path):        
        rep_values = {"weight_vec": self.weight_vec,
                        "centers": self.centers,
                        "widths": self.widths }

        joblib.dump(rep_values, 
                    os.path.join(path, VALUES_FILE))


    def load_from_directory(self, path):
        values_dir = joblib.load(os.path.join(path, VALUES_FILE))
        for attr, value in values_dir.items():
            self.__dict__[attr] = value
            print "Loaded {}".format(attr)
