""" Import binding affinities. """

from os.path import join, dirname
import pandas as pd


path_here = dirname(dirname(__file__))


def human_affinity():
    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
