"""
Runs polyfc to create initial guesses for abundance and Ka for each receptor. 
Polyfc predicted output is compared to measured Atyeo data. 
Cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = ??
"""
# %%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from model import lBnd
import pickle
from scipy.stats import pearsonr

# %%
def initial_AbundKa(cube, n_ab=1):
    """
    Generate abundance and Ka matrices from random values
    cube.shape == n_subj * n_rec * n_Ag
    """
    
    R_subj_guess = np.random.lognormal(size=(cube.shape[0], n_ab))
    R_Ag_guess = np.random.lognormal(size=(cube.shape[2], n_ab))
    Ka_guess = np.random.lognormal(11, size=(cube.shape[1], n_ab))
    print(Ka_guess)
    return R_subj_guess, R_Ag_guess, np.log(Ka_guess)


def infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12):
    """
    Pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Lbound_cube = np.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    # Lbound_guess = 6x1638
    Ka = np.exp(Ka[:, np.newaxis])
    RR = np.einsum("ij,kj->ijk", R_subj, R_Ag)

    for jj in range(Lbound_cube.shape[1]):
        Lbound_cube[:, jj, :] = lBnd(L0, KxStar, RR, Ka[jj, :])

    return Lbound_cube


def model_lossfunc(x, cube, L0=1e-9, KxStar=1e-12):
    """
        Loss function, comparing model output and flattened tensor
    """
    # Unflatten to three matrices
    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(x) / np.sum(cube.shape))

    R_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = x[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    Ka = x[(n_subj + n_Ag) * n_ab:(n_subj + n_Ag + n_rec) * n_ab].reshape(n_rec, n_ab)

    Lbound = infer_Lbound(R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar)
    model_loss = np.nansum((Lbound - cube)**2)
    print(np.isnan(cube.any()))
    
    with open("atyeo_modelloss.pkl", "wb") as output_file:
        pickle.dump(model_loss, output_file)

    all_modelloss = []
    with (open("atyeo_modelloss.pkl", "rb")) as openfile:
        while True:
            try:

                all_modelloss.append(pickle.load(openfile))
            except EOFError:
                break

    print("Model loss: ", model_loss)
    return model_loss


def optimize_lossfunc(cube, n_ab=1, maxiter=100):
    """
        Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))
    bnds = ((0, np.inf), ) * len(x0)

    opt = minimize(model_lossfunc, x0, args=(cube, 1e-9, 1e-12), bounds=bnds, options={"maxiter": maxiter})

    RKa_opt = opt.x[:, np.newaxis]
    return RKa_opt

# %%
def compare(RKa_opt, cube, rec_names, ant_names):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar
    """

    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(RKa_opt) / np.sum(cube.shape))

    R_subj = RKa_opt[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = RKa_opt[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    Ka = RKa_opt[(n_subj + n_Ag) * n_ab:(n_subj + n_Ag + n_rec) * n_ab].reshape(n_rec, n_ab)
    Lbound_model = infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12)


    coeff = np.zeros([n_rec,n_Ag])
    for ii in range(cube.shape[1]):
        for jj in range(cube.shape[2]):
            
            coeff[ii,jj], _ = pearsonr(cube[:,ii,jj],Lbound_model[:,ii,jj])
    
    print(coeff)  

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig = ax.imshow(coeff)  
    

    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(ant_names)),ant_names)
    plt.yticks(np.arange(len(rec_names)),rec_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ant_names)):
        for j in range(len(rec_names)):
            text = ax.text(i, j, round(coeff[j, i],2),
                ha="center", va="center", color="w")
               


    plt.show() 
    return coeff       
    

# Obtain optimal parameters
#%%
from data.atyeo import createCube, getAxes
cube = createCube()
_,rec_names,ant_names = getAxes()


RKa_opt = optimize_lossfunc(cube, n_ab=1, maxiter=100)

with open("atyeo_optparams.pkl", "wb") as output_file:
    pickle.dump(RKa_opt, output_file)


# Generate comparison heat map 
#%%
from data.atyeo import createCube, getAxes
cube = createCube()
_,rec_names,ant_names = getAxes()


with open("atyeo_optparams.pkl", "rb") as open_file:
    optparams = pickle.load(open_file)

optparams = (np.asarray(optparams)).flatten()


coeff = compare(optparams, cube, rec_names, ant_names)

#%%
