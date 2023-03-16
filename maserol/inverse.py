import numpy as np
import seaborn as sns

from .core import *


def inverse(metric="mean_direct", noise = 0.5):
    L0 = 1e-9
    KxStar = 1e-12
    ab_types = ["IgG1", "IgG2", "IgG3"]
    fcr_types = ["FcgRI", "FcgRIIA"]

    Rtot_np = np.random.rand(3000, len(ab_types), 5)
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
    cube.values = inferLbound(cube.values, Rtot.values, Ka.values, lrank=False, L0=L0, KxStar=KxStar, FcIdx=len(ab_types))
    # add a Gaussian noise to measurements
    cube.values = cube + cube * np.random.randn(*cube.shape) * noise

    x, _ = optimizeLoss(cube, metric=metric, lrank=False, fitKa=False, ab_types=tuple(ab_types), L0=L0, KxStar=KxStar)
    x = x[:len(Rtot_np.ravel())]
    return Rtot, x


def plotInverse(Rtot: xr.DataArray, x):
    Rtot_df = Rtot.to_dataframe("Abundance").reset_index()
    ax = sns.scatterplot(x=np.log10(Rtot.values.flatten()), y=np.log10(np.exp(x)),
                         hue=Rtot_df["Antibody"], style=Rtot_df["Antigen"])
    ax.set_title("Rtot: Truth vs Forward-Backward (Log10) (per-receptor scaling factor; noise=0.5)")
    ax.set_xlabel("Log(True Rtot)")
    ax.set_ylabel("Log(Forward-Backward Rtot)")
    return ax