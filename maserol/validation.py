import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

from .core import reshapeParams, optimizeLoss

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

def bootstrap(cube : xr.DataArray, numResample=10, **opt_kwargs):
    '''
    Runs bootstrapping algorithm on MTD 'numResample' times.

    Args:
        cube: DataArray object with processed data
        num_resample: number of times to run bootstrapping
        param_dict: kwargs that are passed into optimizeLoss
    
    Returns:
        [[samples mean, samples std], [ag mean, ag std]] or [abundance mean, abundance std]
    '''
    if opt_kwargs['lrank']:
        subjects_list, ag_list = [], []
    else:
        abundance_list = []

    for _ in range(numResample):
        data = resample(cube)
        x, _ = optimizeLoss(data, **opt_kwargs)
        x = reshapeParams(x, data, opt_kwargs['lrank'], opt_kwargs['fitKa'])

        if (opt_kwargs['lrank']):
            subjects_list.append(x[0])
            ag_list.append(x[1])
        else:
            abundance_list.append(x[0])
    mean_std = lambda l : (np.mean(np.array(l), axis=0), np.std(np.array(l), axis=0))
    return mean_std(subjects_list), mean_std(ag_list) if opt_kwargs['lrank'] else mean_std(abundance_list)

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
