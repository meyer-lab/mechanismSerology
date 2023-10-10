import numpy as np
from tensordata.zohar import data as zohar

from maserol.preprocess import assemble_options, get_kaplonek_mgh_data
from maserol.scatterplot import plot_LLigO


def makeFigure():
    data = get_kaplonek_mgh_data()
    data = data[np.random.choice(data.shape[0], 300)]
    opt_kwargs = assemble_options(data, rcps=["IgG1", "IgG1f", "IgG3", "IgG3f"])
    return plot_LLigO(data, "FcR3A", **opt_kwargs)[0]
