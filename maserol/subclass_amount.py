import numpy as np
import seaborn as sns

from maserol.core import optimizeLoss, reshapeParams

def plot_ab_aggs(cube, abs, runs=6, ax=None):
    abundance_list = []
    for _ in range(runs):
        x_opt, _ = optimizeLoss(cube, lrank=False, ab_types=abs)
        abundance, = reshapeParams(x_opt, cube, lrank=False, ab_types=abs, as_xarray=True)
        abundance_list.append(abundance)
    abundance = np.mean(np.array(abundance_list), axis=0)
    ab_abunds = np.sum(np.sum(abundance, axis=2), axis=0)
    err = np.std(np.sum(np.sum(np.array(abundance_list), axis=3), axis=1), axis=0)
    frac_scalar = 1 / (np.sum(ab_abunds)) * 100
    fracs = ab_abunds * frac_scalar
    fracs_err = err * frac_scalar
    f = sns.barplot(x=np.arange(fracs.size), y=fracs, yerr=fracs_err, ax=ax)
    f.set_title("Average Relative Antibody Amounts")
    f.set_xticklabels(abs)
    f.set_ylabel("Antibody Abundance (%)")
    return abundance_list