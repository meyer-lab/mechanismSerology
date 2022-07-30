from mechanismSerology.maserol.model import prepare_data
import mechanismSerology.maserol.fixkav_opt_helpers as helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr 
import opt_common as oc
import jax.numpy as jnp

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35

def getKavSetup(figsize, gridd, multz=None, empts=None):
    """ Establish figure set-up with subplots. """
    sns.set(style="darkgrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = matplotlib.gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x: x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)

def configure_heatmap(data, title, color, loc):
    """
    Configures settings for and creates heatmap for make_triple_plot.
    """ 
    f = sns.heatmap(data, cmap=color, ax=loc)
    f.set_xticklabels(helpers.absf, rotation=0)
    f.set_xlabel("Antibodies", fontsize=11, rotation=0)
    f.set_title(title, fontsize=13)
    return f

def make_triple_plot(name, subj, ag, kav, n_ab):
    """
    Creates three heatmaps in one plot (Subjects Matrix, Antigen Matrix, Kav Matrix).
    """
    # prepare data
    kav_log = np.log(kav)
    subj_norm, ag_norm = helpers.normalize_subj_ag(subj, ag, n_ab)
    axs, f = getKavSetup((16,6),(1,3))
    plt.subplots_adjust(wspace=.4)

    # plot
    subj_fig = configure_heatmap(subj_norm, "Subjects", "PuBuGn", axs[0])
    ag_fig = configure_heatmap(ag_norm, "Antigens", "PuBuGn", axs[1])
    af_fig = configure_heatmap(kav_log, "Affinities (1/M)", "PuBuGn", axs[2])
    add_triple_plot_labels(name, subj_fig, ag_fig, subj, ag)

    af_fig.set_yticklabels(helpers.affinities_dict[name], fontsize=10, rotation=0)
    f.suptitle(f'{name}', fontsize=18)
    return f

def add_triple_plot_labels(name, subj_fig, ag_fig, subj=None, ag=None):
    """
    Adds labels for specific datasets for make_triple_plot figure.
    """
    if (name == "zohar"):
            outcomes, values = helpers.zohar_patients_labels()
            sum = 0
            for i in range(len(values)):
                original = values[i]
                values[i] = sum
                sum+=original
            subj_fig.set_yticks(values, outcomes, fontsize=8)

    if (name == "alter"):
        ag_fig.set_yticklabels(helpers.antigen_dict[name], fontsize=8, rotation=0)
    else:
        ag_fig.set_yticklabels(helpers.antigen_dict[name], fontsize=10, rotation=0)
    
    if (name == 'atyeo'):
        outcomes = helpers.atyeo_patient_labels()
        subj = pd.DataFrame(subj, columns=helpers.absf)
        subj['Outcomes'] = outcomes
        subj = subj.sort_values('Outcomes')
        subj_fig.set_yticks([0,len(subj['Outcomes'][subj["Outcomes"] == 0.0])], ['Deceased', 'Convalescent'], rotation=0, va='center')


def configure_scatterplot(data : xr.DataArray, lbound, loc=None): 
    """
    Configures settings and creates scatterplot for make_initial_final_lbound_correlation_plot.
    """
    # prepare data
    cube_flat = (prepare_data(data)).values.flatten()
    nonzero = np.nonzero(cube_flat)
    receptor_labels, antigen_labels = helpers.make_rec_subj_labels(data)

    lbound_flat = lbound.flatten()[nonzero]
    cube_flat = cube_flat[nonzero]
    receptor_labels = receptor_labels[nonzero]
    antigen_labels = antigen_labels[nonzero]

    # plot
    f = sns.scatterplot(x=np.log(lbound_flat), y=np.log(cube_flat), hue=receptor_labels, style=antigen_labels, ax=loc, s=70, alpha=0.5)
    f.legend(title="Receptor | Antigen", bbox_to_anchor=(1, 1), borderaxespad=0)
    f.set_xlabel("Predictions", fontsize=12)
    f.set_ylabel("Actual", fontsize=12)
    return f

def make_initial_final_lbound_correlation_plot(cube, initial_lbound, final_lbound, per_receptor):
    """
    Creates two scatterplots comparing correlations between the cube and the initial and final lbound predictions.
    """
    cube = prepare_data(cube)
    axs, f = getKavSetup((13, 5), (1, 2))
    initial_f = configure_scatterplot(cube, initial_lbound, axs[0])
    initial_f.set_title("Initial", fontsize=13)
    final_f = configure_scatterplot(cube, final_lbound, axs[1])
    final_f.set_title("After Abundance Fit", fontsize=13)
    add_r_text(cube, initial_lbound, final_lbound, per_receptor, f)
    return f

def add_r_text(cube, initial_lbound, final_lbound, per_receptor, f):
    """
    Prints r value of each receptor and the average r value on the plot f.
    """
    cube_flat = (prepare_data(cube)).values.flatten()
    nonzero = np.nonzero(cube_flat)
    lbound_flat_initial = initial_lbound.flatten()[nonzero]
    lbound_flat_final = final_lbound.flatten()[nonzero]

    receptor_labels, ag_labels = helpers.make_rec_subj_labels(cube)
    r_index_list = helpers.get_indices(cube, per_receptor) if per_receptor else helpers.get_indices(cube, per_receptor)
    labels = receptor_labels if per_receptor else ag_labels

    initial_r = helpers.calculate_r_list_from_index(cube_flat, lbound_flat_initial, r_index_list, True)
    final_r = helpers.calculate_r_list_from_index(cube_flat, lbound_flat_final, r_index_list, True)
    r_initial = jnp.corrcoef(cube_flat, lbound_flat_initial) [0,1]
    r_final = jnp.corrcoef(cube_flat, lbound_flat_final) [0,1]
    
    # initial
    start = 0.78
    for i in range(len(np.unique(labels[nonzero]))):
        f.text(0.05, start, '$r_{' + np.unique(labels[nonzero])[i] + '}$' + r'= {:.2f}'.format(initial_r[i]), fontsize=12)
        start -=.03
    f.text(0.05, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(initial_r)/len(initial_r)), fontsize=12)
    f.text(0.05, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_initial), fontsize=12)

    # final
    start = 0.78
    for i in range(len(np.unique(labels[nonzero]))):
        f.text(0.55, start, '$r_{' + np.unique(labels[nonzero])[i] + '}$' + r'= {:.2f}'.format(final_r[i]), fontsize=12)
        start -=.03
    f.text(0.55, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(final_r)/len(final_r)), fontsize=12)
    f.text(0.55, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_final), fontsize=12)