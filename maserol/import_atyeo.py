from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))

def getAxes():
    """ Get each of the axes over which the data is measured. """
    df = pd.read_csv("data/atyeo2020/atyeo_covid.csv")
    df = df.filter(regex='SampleID|Ig|Fc|SNA|RCA', axis=1)

    axes = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    axes = axes.columns.str.split(" ", expand = True)

    subject = df['SampleID']
    subject = subject[0:22]

    antigen = []
    receptor = []

    for row in axes:
        if (row[0] not in antigen):
            antigen.append(row[0])
        if (row[1] not in receptor):
            receptor.append(row[1])

    return subject, receptor, antigen



def createCube():
    """ Import the data and assemble the antigen cube. """
    subject, receptor, antigen = getAxes()
    cube = np.full([len(subject), len(receptor), len(antigen)], np.nan)
    
    df = pd.read_csv("data/atyeo2020/atyeo_covid.csv")
    df = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    df = df[0:len(subject)]

    for i, row in df.iterrows():
        for j in range(len(receptor)):
            rec =  df.filter(regex=receptor[j])
            cube[i,j] = rec.iloc[i,:]

    return cube 