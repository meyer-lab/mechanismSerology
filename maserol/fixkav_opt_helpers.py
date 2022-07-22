from tensordata.atyeo import load_file
from tensordata.zohar import pbsSubtractOriginal
import jax.numpy as jnp
import numpy as np
import xarray as xr

abs = ['IgG1', 'IgG1f', 'IgG2', 'IgG2f', 'IgG3', 'IgG3f', 'IgG4', 'IgG4f']

affinities_dict = {
    'alter' : ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'IgG'],
    'atyeo' : ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcRg2A', 'FcRg2b', 'FcRg3A'],
    'kaplonek_mgh' : ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcR2A', 'FcR2B', 'FcR3A', 'FcR3B'],
    'kaplonek_spacex' : ['IgG1', 'IgG3', 'FcR2A', 'FcR3A'],
    'zohar' : ['IgG1', 'IgG2', 'IgG3', 'FcR2A', 'FcR2B', 'FcR3A', 'FcR3B'],
}

antigen_dict = {
    'alter' : ['93TH975.gp120', 'Chiang.Mai.gp120', 'gp120.BAL', 'gp120.BAL.Kif', 'gp120.CM', 'gp120.CM235', 'gp120.CM244', 'gp120.CN54', 
              'gp120.Du151', 'gp120.Du156.12', 'gp120.IIIb', 'gp120.JRCSF', 'gp120.MN', 'gp120.PVO', 'gp120.RSC3', 'gp120.RSC3.delta3711',
              'gp120.SF162', 'gp120.TRO', 'gp120.YU2', 'gp120.ZM109F', 'gp120.96ZM651', 'gp140.BR29', 'gp140.Clade.B', 'gp140.CN54', 'gp140.Du151',
              'gp140.HXBc2', 'gp140.UG21', 'SOSIP', 'gp41.HXBc2', 'gp41.MN', 'HIV1.Integrase', 'HIV1.Nef', 'HIV1.Rev', 'HIV1.Gag', 'HIV1.Vif', 'HIV1.p7', 
              'IIIb.pr55.Gag', 'p24.HXBc2', 'p24.IIIb', 'p51.HIV1.RT', '6H.HIV1.p66'],
    'atyeo' : ['S', 'RBD', 'N'],
    'kaplonek_mgh' : ['SARS.CoV2_N', 'CoV_OC43', 'influenza_HA', 'SARS.CoV.2_S1','Ebola_gp', 'CMV', 'SARS.CoV.2_S', 'SARS.CoV.2_S2','SARS.CoV.2_RBD'] ,
    'kaplonek_spacex' : ['HKU1_S', 'OC43_RBD', 'HKU1_RBD', 'OC43_S', 'CoV2_S', 'CoV2_S1', 'CoV2_RBD', 'SARS_RBD', 'CoV2_S2', 'Flu_HA', 'Ebola', '_MERS_RBD',
                        'SARS_S', 'MERS_S'],
    'zohar' : ['S', 'RBD', 'N', 'S1', 'S2', 'S1 Trimer']
}

def normalize_subj_ag_whole(subj, ag): 
    """
    Normalizes entire antigen matrix.
    """
    max = ag.max()
    ag /= max
    subj *= max
    return subj, ag

def zohar_patients_labels(): 
    """
    Returns a list of Zohar patient outcomes and a list of the number of patients for each outcome.
    """
    df = pbsSubtractOriginal()
    outcome_labels = list(df["group"].unique())
    outcome_values = list(df["group"].value_counts(sort=False))
    return outcome_labels, outcome_values

def atyeo_patient_labels(): 
    """
    Returns a list of outcomes for all Atyeo patients.
    """
    df = load_file("atyeo_covid")
    return list(df["Outcome"][0:22])

def get_receptor_indices(cube : xr.DataArray):
    """
    Returns a matrix of indices where each receptor occurs in a given cube of data.
    """
    receptor_labels, _ = make_rec_subj_labels(cube)
    nonzero_indices = np.nonzero(jnp.ravel(cube.values))
    receptor_labels = receptor_labels[nonzero_indices]

    r_index_matrix = []
    for receptor in np.unique(receptor_labels):
        r_index_matrix.append(np.where(receptor_labels == receptor))
    return r_index_matrix

def make_rec_subj_labels(data: xr.DataArray): 
    """
    Returns a flattened array of receptor and antigen labels for each element of the cube that is given.
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

def calculate_r_list_from_index(cube_flat, lbound_flat, index_matrix):
    """
    Returns a list of r values for every receptor in the cube and lbound at the given path
    """
    r_list = []

    for indices in index_matrix:
            cube_val = cube_flat[indices]
            lbound_val = lbound_flat[indices]
            corr_matrix = jnp.corrcoef(jnp.log(cube_val), jnp.log(lbound_val))
            r = corr_matrix[0,1]
            r_list.append(r)
    return r_list


