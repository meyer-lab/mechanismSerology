import numpy as np
from .figures.common import *
from matplotlib.tri import Triangulation
import xarray as xr
from .preprocess import HIgGs, HIgGFs
from .core import *

def plotOneHeatmap(data, title, color, abs, ax, annot=False):
    """
    Configures settings for and creates heatmap for plotHeatmaps().
    """ 
    f = sns.heatmap(data, cmap=color, ax=ax, annot=annot, annot_kws={'rotation': 90})
    f.set_xticklabels(abs, rotation=90)
    f.set_xlabel("Antibodies", fontsize=11, rotation=0)
    f.set_title(title, fontsize=13)
    return f

def plotHeatmaps(cube: xr.DataArray, x_opt, fitKa=DEFAULT_FIT_KA_VAL, lrank=DEFAULT_LRANK_VAL,
                 outcomes=None, name="", normPerAg=False):
    """
    Creates three heatmaps in one plot (Samples, Antigens, Kav).
    """
    # extract optimization results
    assert lrank, "plotHeatmaps() can only handle low rank-ed abundance (as two matrices)"
    cube = prepare_data(cube)
    opt_ps = reshapeParams(x_opt, cube, lrank=True, fitKa=fitKa)
    if fitKa:
        samp, ag, kav = opt_ps[0], opt_ps[1], opt_ps[2]
    else:
        samp, ag = opt_ps[0], opt_ps[1]
        kav = assembleKav(cube)

    # preprocess data
    kav_log = np.log(kav)
    n_ab = samp.shape[1]
    abs = ["Ab"+str(i) for i in range(1, n_ab+1)]   # default Ag label as "Abx"
    if n_ab == 4:
        abs = HIgGs
    if n_ab == 8:
        abs = HIgGFs
    # normalize Ag matrix and shift weights to samp matrix, may do by each Ag
    agmax = np.max(ag, axis=(0 if normPerAg else None))
    ag /= agmax
    samp *= agmax

    axs, f = getSetup((16,6),(1,3))
    plt.subplots_adjust(wspace=.4)

    # plot
    subj_fig = plotOneHeatmap(samp, "Subjects Log10 ", "PuBuGn", abs, axs[0])
    ag_fig = plotOneHeatmap(ag, "Antigens", "PuBuGn", abs, axs[1])
    af_fig = plotOneHeatmap(kav_log, "Affinities (1/M)", "PuBuGn", abs, axs[2], True)

    # label axes
    ag_fig.set_yticks([x for x in range(len(cube.Antigen))], cube.Antigen.values, fontsize=8, rotation=0)
    af_fig.set_yticklabels(cube.Receptor.values, fontsize=8, rotation=0)
    if (outcomes != None):
        outcomes.sort()
        labels = set(outcomes)
        outcome_index = [outcomes.index(outcome) for outcome in labels]
        subj_fig.set_yticks(outcome_index, labels, fontsize=8, rotation=0)
    f.suptitle(f'{name.capitalize()}', fontsize=18)
    return f, subj_fig, ag_fig, af_fig

def plot_deviation_heatmap(mean_matrix, std_matrix, absf, ylabels):
    '''
    Creates a split-triangle heatmap summarizing MTD bootstrapping results. 
    Inputs:
        mean_matrix, std_matrix: 2D matrices of mean and standard deviation of bootstrapped calues
        absf: names of antibodies
        ylabels: list of ylabels for heatmap
    '''
    lower_map = np.asarray([mean_matrix[i] - std_matrix[i] for i in range(len(mean_matrix))])
    upper_map = np.asarray([mean_matrix[i] + std_matrix[i] for i in range(len(mean_matrix))])
    M = len(absf)
    N = lower_map.shape[0]
    x = np.arange(M + 1)
    y = np.arange(N + 1)
    xs, ys = np.meshgrid(x, y)
    triangles1 = [(i + j*(M+1), i+1 + j*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(M)]
    triangles2 = [(i+1 + j*(M+1), i+1 + (j+1)*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(M)]
    triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
    triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    a = axes.tripcolor(triang1, lower_map.ravel(), vmin=lower_map.min(), vmax=upper_map.max(), edgecolors='white')
    b = axes.tripcolor(triang2, upper_map.ravel(), vmin=lower_map.min(), vmax=upper_map.max(), edgecolors='white')
    fig.colorbar(b)
    plt.xlim(x[0]-0.5, x[-1]-0.5)
    plt.ylim(y[0]-0.5, y[-1]-0.5)
    axes.set_xticks(range(0, len(absf)), absf)
    axes.set_yticks(range(0, len(ylabels)), ylabels)
    return fig

def plot_3D_heatmap(cube : xr.DataArray):
    '''
    Creates 3D heatmap visualization for data in 'cube'.
    '''

    sub = []
    z = cube.values.transpose((2,0,1)).ravel()
    for i in range(len(cube.Antigen)):
        sub.append(cube[:,:,i])
    f = plt.figure(figsize=(10,10))
    a,b,c = np.asarray(sub).nonzero()
    ax = f.add_subplot(111, projection='3d')
    img = ax.scatter3D(b,a,c, c=z, marker='s', s=400, cmap='PuBuGn')
    
    ax.set_xlabel("Subjects", fontsize=20)
    
    label = list(cube.Antigen.values)
    label.insert(0,'Rand')
    ax.set_yticklabels(label)
    ax.set_ylabel("Antigens", fontsize=18)
    
    label = list(cube.Receptor.values)
    label.insert(0,'Rand')
    ax.set_zticklabels(label)
    ax.set_zlabel("Receptors", fontsize=18)

    return f

def separate_weights(matrix, r):
    '''
    Normalizes 'matrix' by dividing each column by its highest value and returns normalized matrix and weights.
    'r' is the rank (number of antibodies)
    '''
    weights = []
    norm_matrix = []
    for i in range(r):
        weight = matrix[:,i].max()
        weights.append(weight)
        norm_matrix.append(matrix[:,i] / weight)
    norm_matrix = np.asarray(norm_matrix)
    norm_matrix = norm_matrix.reshape(matrix.shape) 
    return norm_matrix, weights

def removed_weights_subj_ag(cube, subjects, antigens, absf):
    '''
    Plots 'subjects' and 'antigens' matrices with separated weights.
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'height_ratios': [1, 5]}, figsize=(10,12))
    axes[0,1].axis('off')
    cbar_ax = fig.add_axes([1, .047, .03, .72])

    # get weights and normalized matrices
    snorm, sweights = separate_weights(subjects, len(absf))
    anorm, aweights = separate_weights(antigens, len(absf))
    comb_weights = np.asarray([sum(value) for value in zip(sweights, aweights)])

    # subjects heatmap
    s = sns.heatmap(snorm, cbar=True, cbar_ax=cbar_ax, cmap="PuBuGn", ax=axes[1,0])
    s.set_xticklabels(absf, rotation=45)
    s.set_title("Subjects", fontsize=15)

    # antigens heatmap
    a = sns.heatmap(anorm, cbar=False, cmap="PuBuGn", ax=axes[1,1])
    a.set_yticklabels(cube.Antigen.values, rotation=0)
    a.set_xticklabels(absf, rotation=45)
    a.set_title("Antigens", fontsize=15)

    # weights 
    spal = sns.color_palette("Oranges_r", len(np.asarray(comb_weights)))
    srank = np.asarray(comb_weights).argsort().argsort()
    sb = sns.barplot(y=comb_weights, x=absf, ax=axes[0,0], palette=np.asarray(spal)[::-1][srank])
    sb.set_xticklabels([])
    sb.set_yticklabels([])
    for i in range(len(comb_weights)):
        value = round(comb_weights[i])
        sb.text(i, 1, '{:.2e}'.format(value), fontsize=10, color='black', ha='center', rotation='vertical')

    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    return fig