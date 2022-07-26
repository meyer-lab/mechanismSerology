from tensordata.kaplonek import MGH, MGH4D, SpaceX
from tensordata.zohar import data3D
import numpy as np

def checkMissingess(cube):
    return 1 - np.sum(np.isfinite(cube))/np.product(cube.shape)


def normalizeSubj(cube):
    cube -= np.nanmean(cube, axis=0)
    cube = cube / np.nanstd(cube, axis=0)
    return cube


""" Set consistent naming between the three DataArrays """
def rename(M, S, Z):

    #Specific to Kaplonek MGH 3D/4D
    if (len(M.dims) == 4):
        M = M.rename({'Subject': 'Sample','Day': 'Time'})
    
    M_dict = {'Antigen': ['SARS.CoV2_N', 'CoV.OC43', 'Flu_HA', 'SARS.CoV2_S1', 'Ebola_gp', 'CMV',
                                        'SARS.CoV2_S', 'SARS.CoV2_S2', 'SARS.CoV2_RBD']}

    #Specific to Kaplonek SpaceX
    S_dict = {'Antigen': ['CoV.HKU1_S', 'CoV.OC43_RBD', 'CoV.HKU1_RBD', 'CoV.OC43_S', 'SARS.CoV2_S',
                            'SARS.CoV2_S1', 'SARS.CoV2_RBD', 'SARS_RBD', 'SARS.CoV2_S2', 'Flu_HA',
                            'Ebola_gp', 'MERS_RBD', 'SARS_S', 'MERS_S'],
                'Sample': S['Sample'].values.astype(str)}

    #Specific to Zohar
    Z_dict = {'Antigen': ['SARS.CoV2_S', 'SARS.CoV2_RBD', 'SARS.CoV2_N', 'SARS.CoV2_S1','SARS.CoV2_S2', 
                            'SARS.CoV2_S1trimer']}

    return M.assign_coords(M_dict), S.assign_coords(S_dict), Z.assign_coords(Z_dict)

    

""" Assemble the concatenated COVID tensor in 3D """
def importConcat():
    M, S, Z = MGH(xarray=True), SpaceX(xarray=True), data3D(xarray=True)

    M = normalizeSubj(M)
    S = normalizeSubj(S)
    Z = normalizeSubj(Z)

    M, S, Z = rename(M, S, Z)
    cube = S.combine_first(M).combine_first(Z)
    cube = normalizeSubj(cube)

    return cube, M, S, Z


def importConcat4D():
    M, S, Z = MGH4D(xarray=True), SpaceX(xarray=True), data3D(xarray=True)

    M = normalizeSubj(M)
    S = normalizeSubj(S)
    Z = normalizeSubj(Z)

    M, S, Z = rename(M, S, Z)

    S_time = np.arange(S['Sample'].size)
    S = S.expand_dims({'Time': S['Sample'].size}).assign_coords({'Time': S_time})

    Z_time = np.arange((Z['Sample'].size))
    Z = Z.expand_dims({'Time': Z['Sample'].size}).assign_coords({'Time': Z_time})
   
    concat = S.combine_first(M).combine_first(Z)
    concat = normalizeSubj(concat)

    return concat, M, S, Z

