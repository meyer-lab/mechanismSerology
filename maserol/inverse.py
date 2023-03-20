import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

from .core import *
from .figures.common import getSetup


def genForwardSim(noise = 0.5, n_sample=1000, n_antigen=5):
    L0 = 1e-9
    KxStar = 1e-12
    ab_types = ["IgG1", "IgG2", "IgG3"]
    fcr_types = ["FcgRI", "FcgRIIA"]

    Rtot_np = np.random.rand(n_sample, len(ab_types), n_antigen)
    log_steps = 9
    idx_stepsize = int(Rtot_np.shape[0] / log_steps)
    for i in range(log_steps):
        Rtot_np[i * idx_stepsize:(i + 1) * idx_stepsize, :, :] *= 10 ** i

    Rtot = xr.DataArray(Rtot_np,
                        [np.arange(Rtot_np.shape[0]), ab_types, np.arange(Rtot_np.shape[2])],
                        ["Sample", "Antibody", "Antigen"], )
    cube = xr.DataArray(np.zeros((Rtot.shape[0], len(ab_types + fcr_types), Rtot.shape[2])),
                        [Rtot.Sample.values, ab_types + fcr_types, Rtot.Antigen.values],
                        ["Sample", "Receptor", "Antigen"])

    Ka = assembleKav(cube, tuple(ab_types))
    cube.values = inferLbound(cube.values, Rtot.values, Ka.values, lrank=False, L0=L0, KxStar=KxStar,
                              FcIdx=len(ab_types))
    # add a Gaussian noise to measurements
    cube.values = cube + cube * np.random.randn(*cube.shape) * noise
    return cube, Rtot


def plotInverse():
    cube, Rtot = genForwardSim()
    ab_types = Rtot.Antibody.values
    L0, KxStar = 1e-9, 1e-12

    ax, f = getSetup((12, 4), (1, 3))
    metrics = ["mean_direct", "mean", "mean_rcp"]
    captions = ["no scaling factors", "single scaling factor", "per-receptor scaling factor"]

    for i in range(len(metrics)):
        x, _ = optimizeLoss(cube, metric=metrics[i], lrank=False, fitKa=False, ab_types=tuple(ab_types), L0=L0,
                            KxStar=KxStar)
        x = x[:np.prod(Rtot.shape)]

        Rtot_df = Rtot.to_dataframe("Abundance").reset_index()
        sns.scatterplot(ax=ax[i], x=np.log10(Rtot.values.flatten()), y=np.log10(np.exp(x)),
                             hue=Rtot_df["Antibody"], style=Rtot_df["Antigen"])
        ax[i].set_title(f"Rtot: Truth vs Forward-Backward (Log10) ({captions[i]})")
        ax[i].set_xlabel("Log(True Rtot)")
        ax[i].set_ylabel("Log(Forward-Backward Rtot)")
    return f
