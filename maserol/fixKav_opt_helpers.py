import pickle
import numpy as np
import maserol.fixKav_optimization as fk
import maserol.model as model

def run_optimization(cube, use_r=False, abs=4):
    cube = model.prepare_data(cube)
    kav = model.assemble_Kav(cube)
    kav[np.where(kav==0.0)] = 10
    kav_log = np.log(kav)
    if (use_r):
        final_matrix_r2 = fk.optimize_lossfunc(cube.values, kav_log.values, True, 4)
        r_subj_pred_r, r_ag_pred_r = fk.reshapeParams(final_matrix_r2, cube)
        lbound = fk.infer_Lbound(r_subj_pred_r, r_ag_pred_r, kav_log.values)
        return r_subj_pred_r, r_ag_pred_r, lbound
    else:
        final_matrix_diff = fk.optimize_lossfunc(cube.values, kav_log.values, False, 4)
        r_subj_pred_d, r_ag_pred_d = fk.reshapeParams(final_matrix_diff, cube)
        lbound = fk.infer_Lbound(r_subj_pred_d, r_ag_pred_d, kav_log.values)
        return r_subj_pred_d, r_ag_pred_d, lbound

def pickle_wrapper(name, item, data, is_r):
    if (is_r):
        with open(f"./data/{name}_{item}_r.pickle", 'wb') as f:
            pickle.dump(data, f)
        f.close()
    else:
        with open(f"./data/{name}_{item}.pickle", 'wb') as f:
            pickle.dump(data, f)
        f.close()
    

def save_optimization_data_r(name, rsr, rar, kav, lbound):
    # using r-squared
    pickle_wrapper(name, "subjects", rsr, True)
    pickle_wrapper(name, "ag", rar, True)

    # affinities matrix
    pickle_wrapper(name, "affinities", kav, False)

    # lbound
    pickle_wrapper(name, "lbound", lbound, True)

def save_optimization_data_d(name, rsd, rad, kav, lbound):
    # using difference
    pickle_wrapper(name, "subjects", rsd, False)
    pickle_wrapper(name, "ag", rad, False)

    # affinities matrix
    pickle_wrapper(name, "affinities", kav, False)

    # lbound
    pickle_wrapper(name, "lbound", lbound, False)