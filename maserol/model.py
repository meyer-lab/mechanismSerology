""" Import binding affinities. """

from os.path import join, dirname
import pandas as pd
import numpy as np


path_here = dirname(dirname(__file__))


def human_affinity():
    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)



def makeAffinities(receptors):
    """ Build affinities matrix given receptors axis """
    humanAff = pd.read_csv("maserol/data/human-affinities.csv",
                            delimiter=",", comment="#", index_col=0)

    #Filter for only IgG1-2 and FcRg2-3, based on naming scheme of dataset
    if any('FcRg' in x for x in receptors):
        rows = ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcRg2A', 'FcRg2b', 'FcRg3A']
    elif any('FcgR' in x for x in receptors):
        rows = ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcgRIIa.H131', 'FcgRIIb', 'FcgRIIIa.V158']
    elif any('FcR' in x for x in receptors):
        rows = ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcR2A', 'FcR2B', 'FcR3A'] 
    else:
        raise Exception("Receptor naming scheme does not match")

    columns = rows[0:4]

    affinities = np.zeros((len(rows), 4))
    affinities[0:4] = np.identity(4) * 10**9
    
    #Add relevant human affinities and select one allele where applicable 
    select = ['FcgRIIA-131H', 'FcgRIIB-232I', 'FcgRIIIA-158V']
    affinities[4:] = humanAff.loc[select]

    return affinities, rows, columns
    
    