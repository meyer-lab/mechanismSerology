import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

from tensordata.alter import load_file, data as alter

from maserol.figures.common import getSetup
from maserol.preprocess import HIgGFs, prepare_data
from maserol.relative import plot_ab_aggs

def makeFigure():
    axes, fig = getSetup((9, 4.5), (1, 2))
    gp120 = load_file("data-glycan-gp120")
    abs = HIgGFs
    cube_alter = prepare_data(alter()["Fc"], data_id="alter")
    gp120_ag = [ag for ag in cube_alter.Antigen.values if "gp120" in ag]
    abundance_list = plot_ab_aggs(cube_alter.sel(Antigen=gp120_ag, Sample=gp120["subject"].values), abs, 3, 650, metric="mean_rcp", ax=axes[0])

    measured_fucose_ratio = gp120["F.total"]
    fucose_abs = [ab for ab in abs if ab.endswith("f")]
    get_aggs = lambda abs: np.sum(np.sum(np.mean(np.array([abund.sel(Antibody=abs) for abund in abundance_list]), axis=0), axis=2), axis=1)
    inferred_fucose_ratio = get_aggs(fucose_abs) / get_aggs(abs) * 100
    f = sns.scatterplot(x=inferred_fucose_ratio, y=measured_fucose_ratio, ax=axes[1])
    f.set_xlabel("Inferred Fucose Ratio (%)")
    f.set_ylabel("Measured Fucose Ratio (%)")
    f.set_title("Inferred vs Measured Fucose Ratios")
    r = np.corrcoef(inferred_fucose_ratio, measured_fucose_ratio)[0,1]
    mse = mean_squared_error(inferred_fucose_ratio, measured_fucose_ratio)
    axes[1].text(0.8, 0.1, f"r={round(r, 2)}\nMSE: {round(mse, 2)}", transform=axes[1].transAxes)
    return fig
