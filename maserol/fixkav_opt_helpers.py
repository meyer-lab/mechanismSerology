import jax.numpy as jnp
import numpy as np
import xarray as xr

def normalize_subj_ag(subj, ag, n_ab, whole=True): 
    """
    Normalizes antigen matrix. If 'whole' is False, normalizes antigen matrix by columns.
    """
    if (whole):
        ag /= ag.max()
        subj *= ag.max()
    else:
        for i in range(n_ab):
            max = ag[:,i].max()
            ag = ag.at[:,i].set(ag[:,i] / max)
            subj = subj.at[:,i].set(subj[:,i] * max)
    return subj, ag

def make_rec_subj_labels(data: xr.DataArray): 
    """
    Returns a flattened array of receptor and antigen labels for each element of data.
    """
    cube_labels = np.zeros((len(data.Sample.values), len(data.Receptor.values), len(data.Antigen.values)), dtype="O")
    for i in range (len(data.Sample.values)):
        for j in range (len(data.Receptor.values)):
            for k in range(len(data.Antigen.values)):
                cube_labels[i][j][k] = [f'{data.Receptor.values[j]}', f'{data.Antigen.values[k]}']
  
    receptor_labels = []
    antigen_labels = []
    for receptor, antigen in cube_labels.flatten():
        receptor_labels.append(receptor)
        antigen_labels.append(antigen)
    return np.array(receptor_labels), np.array(antigen_labels)

def get_indices(data : xr.DataArray, per_receptor=True):
    """
    Returns a matrix of indices where each receptor occurs in data.
    If 'per_receptor' is False, returns a matrix of indices where each antigen occurs in data.
    """
    receptor_labels, ag_labels = make_rec_subj_labels(data)
    labels = (receptor_labels) if per_receptor else (ag_labels)
    nonzero_indices = ~np.isnan(jnp.ravel(data.values))
    labels = labels[nonzero_indices]

    r_index_matrix = []
    for i in np.unique(labels):
        r_index_matrix.append(np.where(labels == i))
    return r_index_matrix

def calculate_r_list_from_index(cube_flat, lbound_flat, index):
    """
    Returns a list of r values for every receptor/antigen in the cube and lbound
    """
    r_list = []
    for indices in index:
        cube_val = cube_flat[indices]
        lbound_val = lbound_flat[indices]
        corr_matrix = jnp.corrcoef(jnp.log10(cube_val), jnp.log10(lbound_val))
        r = corr_matrix[0,1]
        r_list.append(r)
    return r_list


