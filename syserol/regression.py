""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from scipy.stats import pearsonr
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    selectAlter,
)


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False).dropna()

    # separate dataframes
    Y = df[function]
    X = df.drop(["subject"] + functions, axis=1)

    # perform regression
    Y_pred, coef = RegressionHelper(X, Y)
    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def function_prediction(Xin, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    subset = np.isfinite(Y)
    X = Xin[subset, :]  # subjects x components matrix
    Y = Y[subset]

    # Perform Regression
    Y_pred, coef = RegressionHelper(X, Y)
    Y, Y_pred = selectAlter(Y, Y_pred, evaluation, subset=subset)

    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def RegressionHelper(X, Y, classify=False):
    """ Function with the regression cross-validation strategy. """
    kern = ConstantKernel() * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()

    if classify:
        X = scale(X)
        est = LogisticRegressionCV(penalty="elasticnet", solver="saga")
        estG = GaussianProcessClassifier(kern, warm_start=True)
    else:
        est = ElasticNetCV(normalize=True)
        estG = GaussianProcessRegressor(kern, normalize_y=True)

    est.l1_ratios = [0.8]
    est.cv = 10
    est.max_iter = 10000

    est = est.fit(X, Y)
    coef = np.squeeze(est.coef_)

    Y_pred = cross_val_predict(est, X, Y, cv=20, n_jobs=-1)

    if X.shape[1] < 20:
        Y_pred_G = cross_val_predict(estG, X, Y, cv=20, n_jobs=-1)

        if classify:
            better = accuracy_score(Y, Y_pred_G) > accuracy_score(Y, Y_pred)
        else:
            better = pearsonr(Y, Y_pred_G)[0] > pearsonr(Y, Y_pred)[0]

        if better:
            Y_pred = Y_pred_G

    return Y_pred, coef
