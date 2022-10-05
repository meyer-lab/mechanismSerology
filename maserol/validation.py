import numpy as np
import statistics
import xarray as xr
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from .mechanistic import reshapeParams, optimizeLoss

def resample(cube : xr.DataArray):
    '''
    Returns a DataArray with resampled values from the given 'cube'.
    '''
    cube.name = 'cube'
    df = cube.to_dataframe(dim_order=['Sample', 'Receptor', 'Antigen'])
    df.reset_index(inplace=True)
    resampled = df.groupby(['Antigen', 'Receptor'], group_keys=False).apply(lambda x: x.sample(frac=1.0, replace=True))
    cube.values = np.reshape(list(resampled['cube']), (cube.shape[0], cube.shape[1], cube.shape[2]))
    return cube

def bootstrap(cube : xr.DataArray, param_dict, numResample=10):
    '''
    Runs bootstrapping algorithm on MTD 'numResample' times.

    Inputs:
        cube: DataArray object
        param_dict: list of parameters needed to run MTD
            {metric: metric to use for evaluation function ('mean', 'rtot', or 'r'),
            lrank: when True uses low-rank assumption, 
            retKav: when True affinity matrix is also optimized,
            perReceptor: only applicable with 'r' metric, when True r calculated per-receptor, when False r calculated per-antigen,
            absf: list of antibody names
            }
        num_resample: number of times to run bootstrapping
    
    Outputs:
        list of bootstrapped sample matrix, antigen matrix values or abundance matrix values
    '''
    if param_dict['lrank']:
        subjects_list, ag_list = [], []
    else:
        abundance_list = []
    
    for i in range(numResample):
        data = resample(cube)
        x, val = optimizeLoss(data, param_dict['metric'], param_dict['absf'], param_dict['lrank'], \
                              param_dict['retKav'], param_dict['perReceptor'], len(param_dict['absf']))
        x = reshapeParams(x[:-1], data, param_dict['lrank'], param_dict['retKav'])

        if (param_dict['lrank']):
            subjects_list.append(x[0].flatten())
            ag_list.append(x[1].flatten())
        else:
            abundance_list.append(x[0])
    return subjects_list, ag_list if param_dict['lrank'] else abundance_list

def calculate_bootstrap_mean(matrix_flat):
    '''
    Returns mean for each values in list of bootstrapped values given by 'matrix_flat'.
    '''
    means = []
    for i in range(len(matrix_flat[0])):
        values = [item[i] for item in matrix_flat]
        mean = sum(values) / len(values)
        means.append(mean)
    return np.asarray(means)

def calculate_bootstrap_std(matrix_flat):
    '''
    Returns standard deviation for each values in list of bootstrapped values given by 'matrix_flat'.
    '''
    stds = []
    for i in range(len(matrix_flat[0])):
        values = [item[i] for item in matrix_flat]
        std = statistics.stdev(np.asarray(values))
        stds.append(std)
    return np.asarray(stds)

def prepare_lr_data(subjects_matrix, outcomes, absf, classes, norm=None):
    '''
    Prepares training and testing data for regression. Normalizes data using MinMaxScaler().
    '''
    y = outcomes
    subj = pd.DataFrame(subjects_matrix, columns=absf)
    subj['Outcomes'] = y
    subj = subj[subj.Outcomes.isin(classes)]
    subj = subj.sort_values('Outcomes')
    x,y = subj[absf].to_numpy(copy=True), subj['Outcomes'].to_numpy(copy=True)
    train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y, random_state=0, test_size=0.2)
    if norm:
        scaler = MinMaxScaler(feature_range=(0,1))
        norm_train_x = scaler.fit_transform(train_x)
        norm_test_x = scaler.transform(test_x)
    return norm_train_x, norm_test_x, train_y, test_y

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    Helper function that prints out important regression results.
    '''
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

def hyperparameter_tuning(model, grid, train_x, train_y, test_x, test_y, splits=10, repeats=10):
    '''
    Runs automatic hyperparameter tuning on classification models with 'model' and parameters specificed by 'grid'.
    Returns model with best results.
    '''
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv =cv, scoring="accuracy")
    gridResult = gridSearch.fit(train_x, train_y)
    
    print("Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))
    print_score(gridResult.best_estimator_, train_x, train_y, test_x, test_y, train=False)

    return gridResult.best_estimator_  

def get_crossval_info(model, train_x, train_y, splits=10, repeats=10):
    '''
    Crossvalidates regression using 'model'.
    '''
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    return cross_validate(model, train_x, train_y, cv=cv, return_estimator=True, n_jobs=2)
