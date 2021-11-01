import numpy as np
import matplotlib.pyplot as plt

# r = # of antibody types/bead
from valentbind import polyfc
from .import_kaplonek import flattenSpaceX

#first function: generate abundance and Ka matrices from random values

def generate_AbundKa():
    SX_flatCube, SX_subxant_names, SX_unique_rec_names = flattenSpaceX()
    R_guess = np.random.randint(4, high=11, size=(1,SX_flatCube[0,:].size), dtype=int)
    Ka_guess = np.random.randint(1e8,high=9e8, size=(SX_flatCube[:,0].size,1), dtype=int)

    return R_guess, Ka_guess


#second function: pass the matrices generated above into polyfc, run through each receptor and antxsub pair and store in matrix same size as flatten
def infer_Lbound(n_rec, n_samp, n_ab):
    SX_flatCube, SX_subxant_names, SX_unique_rec_names = flattenSpaceX()
    R_guess, Ka_guess = generate_AbundKa()
    assert(R_guess.shape[1] == n_rec)
    assert(Ka_guess.shape[0] == n_samp)

    #Does experimental data contain nan? -> change to 0
    if np.isnan(SX_flatCube).any():
        nan_data = np.isnan(SX_flatCube)
        nan_row,nan_col = np.where(nan_data == 1)
        SX_flatCube = np.where(np.isfinite(SX_flatCube), SX_flatCube, 0)


    Lbound_guess = np.zeros((SX_flatCube.shape))
    L0 = np.array([1e-9])
    KxStar = 1e-12
    f = 2
    LigC = np.array([1])
    for ii in range(Lbound_guess[:,0].size):
        for xx in range(Lbound_guess[0,:].size):
            Lbound, _, _ = polyfc(L0, KxStar, f, R_guess[0,xx], LigC, np.array([Ka_guess[ii,0]]))
            Lbound_guess[ii,xx] = Lbound

    return Lbound_guess, SX_flatCube


#third function loss function comparing model and flattenSpaceX output
def model_lossfunc():
    SX_flatCube, SX_subxant_names, SX_unique_rec_names = flattenSpaceX()

    Lbound_guess, SX_flatCube = infer_Lbound(SX_subxant_names.size,SX_unique_rec_names.size,1)

    model_loss = np.zeros((SX_flatCube.shape))
    for ii in range(Lbound_guess[:,0].size):
        for xx in range(Lbound_guess[0,:].size):
            indiv_loss = (Lbound_guess[ii,xx] - SX_flatCube[ii,xx])**2
            model_loss[ii,xx] = indiv_loss

    model_loss_all = np.sum(model_loss)

    return model_loss, model_loss_all
