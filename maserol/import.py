""" Import binding affinities. """

import pickle
from functools import reduce
from functools import lru_cache
from os.path import join, dirname
import numpy as np
import pandas as pd


path_here = dirname(dirname(__file__))

def human_affinity():
    data = pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"), delimiter=",", comment="#")

    np.genfromtxt(join(path_here, "maserol/data/murine-affinities.csv"),
                  delimiter=',',
                  skip_header=1,
                  usecols=list(range(1, 6)),
                  dtype=np.float64)