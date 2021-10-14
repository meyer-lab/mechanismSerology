from os.path import join, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_here = dirname(dirname(__file__))

#import .read_csv
#organize data samplexantigenxreceptor
#find indices of each receptor and antigen, store
def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "maserol/data/kaplonek2021/" + name + ".csv"), delimiter=",", comment="#")

    return data
ant_col = 0
rec_col = 1
ant_col = 2
rec_col = 3
data_name = 'MGH_Sero.Neut.WHO124.log10'
features_name = 'MGH_Features'
'SpaceX_Sero.Data'
'SpaceX_Features'

def generateCube(data_name,features_name,ant_col,rec_col):

    data = load_file(data_name)
    data = data.values[:,1:]
    ant_rec_names = load_file(features_name)
    ant_names = ant_rec_names.values[:,ant_col]
    rec_names = ant_rec_names.values[:,rec_col]

    if data[0,:].size != rec_names.size:
        function_data = data[:,data[0,:].size:]
        data = data[:,:data[0,:].size]

    _,unique_rec_ind = np.unique(rec_names, return_index =True)
    unique_rec_names = rec_names[sorted(unique_rec_ind)]


    rec_ind = np.zeros((unique_rec_names.size,int(rec_names.size/unique_rec_names.size))).astype(int)

    for xx in range(unique_rec_names.size):
        rec_index = np.where(rec_names == unique_rec_names[xx])
        rec_index = np.array(rec_index)
        rec_ind[xx,:] = rec_index


    data_cube = np.zeros((data[:,0].size,unique_rec_names.size,rec_ind[0,:].size))

    for subject_ind in range(np.size(data_cube,0)):
        for receptor_ind in range(np.size(data_cube,1)):
            data_cube[subject_ind,receptor_ind,:] = data[subject_ind,rec_ind[receptor_ind,:]]

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(data_cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(data_cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(data_cube), axis=(1, 2)))

    return data_cube, data

d_name = 'SpaceX_Sero.Data' #'MGH_Sero.Neut.WHO124.log10'
f_name = 'SpaceX_Features' #'MGH_Features'
antigen_col = 2 #0
receptor_col = 3 #1

data_cube, data = generateCube(d_name,f_name,antigen_col,receptor_col)

#Checking data order (MGH)
#assert(MGH_cube[60,5,7] == data[60,52])
#assert(MGH_cube[0,9,8] == data[0,89] )
#assert(MGH_cube[578,4,0] == data[578,36])

#Checking data order (SpaceX)
assert(SpaceX_cube[116,5,0] == data[116,70])
assert(SpaceX_cube[0,0,13] == data[0,13])
assert(SpaceX_cube[66,3,4] == data[66,46])
