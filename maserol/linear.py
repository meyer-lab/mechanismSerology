from tensordata.kaplonek import MGH, MGH4D, SpaceX
from tensordata.zohar import data3D
import numpy as np

""" Assemble the concatenated COVID tensor in 3D """
def importConcat():
    M, S, Z = MGH(xarray=True), SpaceX(xarray=True), data3D(xarray=True)

    # Set consistent antigen naming
    M_dict, S_dict, Z_dict = {'Antigen': ['SARS.CoV2_N', 'CoV.OC43', 'Flu_HA', 'SARS.CoV2_S1', 'Ebola_gp', 'CMV',
                                          'SARS.CoV2_S', 'SARS.CoV2_S2', 'SARS.CoV2_RBD']}, {
                                 'Antigen': ['CoV.HKU1_S', 'CoV.OC43_RBD', 'CoV.HKU1_RBD', 'CoV.OC43_S', 'SARS.CoV2_S',
                                             'SARS.CoV2_S1', 'SARS.CoV2_RBD', 'SARS_RBD', 'SARS.CoV2_S2', 'Flu_HA',
                                             'Ebola_gp', 'MERS_RBD', 'SARS_S', 'MERS_S'],
                                 'Sample': S['Sample'].values.astype(str)}, {
                                 'Antigen': ['SARS.CoV2_S', 'SARS.CoV2_RBD', 'SARS.CoV2_N', 'SARS.CoV2_S1',
                                             'SARS.CoV2_S2', 'SARS.CoV2_S1trimer']}

    M, S, Z = M.assign_coords(M_dict), S.assign_coords(S_dict), Z.assign_coords(Z_dict)
    combined = S.combine_first(M).combine_first(Z)

    ## TODO: normalize data correctly

    return combined

def importConcat4D():
    pass



