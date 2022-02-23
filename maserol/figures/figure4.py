import numpy as np
from matplotlib import gridspec, pyplot as plt
import pandas as pd
from tensordata.atyeo import createCube, getAxes, load_file
import seaborn as sns
from string import ascii_lowercase
from tensorpack import perform_CP
from ..predictAbundKa import *

def makeFigure():
    """ """
    cube = createCube()
    subjects, antigens, receptors = optimize_lossfunc(cube)

    optCube = np.full([len(subjects), len(receptors), len(antigens)], np.nan)
    
    df = load_file("atyeo_covid")
    df = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    df = df[0:len(subjects)]

    for i, row in df.iterrows():
        for j in range(len(receptors)):
            rec =  df.filter(regex=receptors[j])
            optCube[i,j] = rec.iloc[i,:]
    
    # Check that there are no slices with completely missing data        
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    tfac = perform_CP(tOrig=optCube)
    components =  [str(ii + 1) for ii in range(tfac.rank)]

    subs = pd.DataFrame(tfac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=subjects)
    rec = pd.DataFrame(tfac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=receptors)
    ant = pd.DataFrame(tfac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=antigens)
    
    f = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    sns.heatmap(subs, cmap="PiYG", center=0, xticklabels=components, yticklabels=subjects, cbar=True, vmin=-1.0, vmax=1.0, ax=ax1)
    sns.heatmap(rec, cmap="PiYG", center=0, xticklabels=components, yticklabels=receptors, cbar=False, vmin=-1.0, vmax=1.0, ax=ax2)
    sns.heatmap(ant, cmap="PiYG", center=0, xticklabels=components, yticklabels=antigens, cbar=False, vmin=-1.0, vmax=1.0, ax=ax3)

    ax1.set_xlabel("Components")
    ax1.set_title("Subjects")
    ax2.set_xlabel("Components")
    ax2.set_title("Receptors")
    ax3.set_xlabel("Components")
    ax3.set_title("Antigens")

    return f