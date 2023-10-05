import numpy as np
from tensordata.zohar import data as zohar

from maserol.preprocess import assemble_options, prepare_data
from maserol.scatterplot import plot_LLigO


def makeFigure():
    data = prepare_data(zohar())
    data = data.sel(Ligand=[lig for lig in data.Ligand.values if lig != "IgG2"])
    opt_kwargs = assemble_options(data, rcps=["IgG1", "IgG1f", "IgG3", "IgG3f"])
    return plot_LLigO(
        data[np.random.choice(data.shape[0], 500, replace=False)], "FcR3B", **opt_kwargs
    )[0]
