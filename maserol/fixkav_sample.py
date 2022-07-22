from fixkav_opt import run_optimization
import fixkav_opt_figures as figs
import matplotlib.pyplot as plt
from tensordata.atyeo import data
import model
import numpy as np
from fixkav_opt import initial_subj_abund
from predictAbundKa import infer_Lbound

def main():
    # set up
    cube = data(xarray=True)
    cube = model.prepare_data(cube)
    kav = model.assemble_Kavf(cube)
    kav = kav.loc[dict(Abs=['IgG1', 'IgG1f', 'IgG2', 'IgG2f', 'IgG3', 'IgG3f', 'IgG4'])]
    kav[np.where(kav==0.0)] = 10
    kav_log = np.log(kav)

    # running optimization and making plots
    # using r for loss function
    r_subj_pred, r_ag_pred, lbound, kav = run_optimization(cube, True, 7)
    r_subj_guess, r_ag_guess = initial_subj_abund(cube.values, n_ab=7)
    lbound_initial = infer_Lbound(r_subj_guess, r_ag_guess, kav_log.values)
    f = figs.make_triple_plot('atyeo', r_subj_pred, r_ag_pred, kav)
    f.suptitle("Atyeo")
    plt.savefig('./figures/fix_kav_atyeo_triple_plot.png')

    f = figs.make_initial_final_lbound_correlation_plot(cube, lbound_initial,lbound)
    f.suptitle("Atyeo")
    plt.savefig('./figures/fix_kav_atyeo_correlation.png')

    # using mean sqaured for loss function
    r_subj_pred, r_ag_pred, lbound, kav = run_optimization(cube, False, 7)
    r_subj_guess, r_ag_guess = initial_subj_abund(cube.values, n_ab=7)
    lbound_initial = infer_Lbound(r_subj_guess, r_ag_guess, kav_log.values)

    f = figs.make_triple_plot('atyeo', r_subj_pred, r_ag_pred, kav)
    f.suptitle("Atyeo")
    plt.savefig('./figures/fix_kav_atyeo_triple_plot_mean.png')

    f = figs.make_initial_final_lbound_correlation_plot(cube, lbound_initial,lbound)
    f.suptitle("Atyeo")
    plt.savefig('./figures/fix_kav_atyeo_correlation_mean.png')

if __name__ == '__main__':
    main()
