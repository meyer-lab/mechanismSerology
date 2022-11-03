from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale

from tensordata.zohar import data3D, pbsSubtractOriginal


def regression(x, y, scale_x: Optional[int] = None, l1_ratio=0.7):
    """ 
    Runs regression with cross-validation.

    Args:
        x: array of shape (n_samples * n_features)
        y: array of shape (n_samples, )
        scale_x: if not None, will scale x along the 

    Returns: model predictions, model, x, y
    """
    if scale_x is not None:
        x = scale(x, axis=scale_x)
    cv = KFold(n_splits=10, shuffle=True)
    if y.dtype == int:
        estCV = LogisticRegressionCV(penalty="elasticnet", solver="saga", cv=cv, l1_ratios=[l1_ratio], n_jobs=-1, max_iter=1000000)
        estCV.fit(x, y)
        model = LogisticRegression(C=estCV.C_[0], penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, max_iter=1000000)
    else:
        assert y.dtype == float
        y = scale(y)
        estCV = ElasticNetCV(normalize=True, l1_ratio=l1_ratio, cv=cv, n_jobs=-1, max_iter=1000000)
        estCV.fit(x, y)
        model = ElasticNet(normalize=True, alpha=estCV.alpha_, l1_ratio=l1_ratio, max_iter=1000000)
    model = model.fit(x, y)
    y_pred = cross_val_predict(model, x, y, cv=cv, n_jobs=-1)
    assert np.any(model.coef_) # check if high l1 ratio zeroed the coefficients
    return y_pred, model, x, y

def get_labels_zohar(multiclass=True):
    data_full = pbsSubtractOriginal()
    sample_class = data_full["group"].to_numpy()
    if not multiclass:
        p = "Progressor"
        c = "Controller"
        cons_mappings = {
            "Deceased": p,
            "Severe": p,
            "Moderate": p,
            "Mild": c,
            "Negative": c,
        }
        sample_class = np.array(list(map(lambda c: cons_mappings[c], sample_class)))
    label_encoder = LabelEncoder()
    sample_class_enc = label_encoder.fit_transform(sample_class)
    return sample_class_enc, label_encoder
    
def plot_roc(x, y, model: BaseEstimator, label_encoder: LabelEncoder, ax=None):
    probs = model.predict_proba(x)
    n_classes = probs.shape[1]
    # onehot encode y
    y_onehot = OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    # x and y for the ROC figure
    fpr = np.array([])
    tpr = np.array([])
    labels = np.array([])
    scores = []
    classes = label_encoder.inverse_transform(np.arange(n_classes))
    if len(classes) == 2:
        # if there are two classes, show only one of them
        classes = classes[1:2]
    for c_idx, c in enumerate(classes):
        scores.append(roc_auc_score(y_onehot[:, c_idx], probs[:, c_idx]))
        fpr_c, tpr_c, _ = roc_curve(y_onehot[:, c_idx], probs[:, c_idx])
        fpr = np.append(fpr, fpr_c)
        tpr = np.append(tpr, tpr_c)
        labels = np.append(labels, np.full(fpr_c.shape, c))
    f = sns.lineplot(fpr, tpr, hue=labels, ci=None, ax=ax, palette="bright")
    text = "AUC:\n" + "\n".join([f"{c}: {score.round(2)}" for c, score in zip(classes, scores)])
    f.text(0.6, 0.05, text)
    f.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    return f

def plot_confusion_matrix(x, y, model: BaseEstimator, label_encoder: LabelEncoder, ax=None):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred, normalize="pred")
    labels = label_encoder.inverse_transform(np.arange(cm.shape[0]))
    f = sns.heatmap(cm, xticklabels=labels, yticklabels=labels, ax=ax)
    return f

def plot_regression_weights(model, ab_types, ax=None):
    """
    Plots regression weighs for each component when using logistic regression.
    """
    coefs = np.squeeze(model.coef_)
    f = sns.barplot(list(ab_types), coefs, palette="bright", ax=ax)
    f.axhline(0, color='k', clip_on=False, linestyle='--')
    f.set(xlabel="Component", ylabel="Component Weight")
    sns.despine(left=False, bottom=False)
    return f

def get_crossval_info(model, train_x, train_y, splits=10, repeats=10):
    '''
    Crossvalidates regression using 'model'.
    '''
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    return cross_validate(model, train_x, train_y, cv=cv, return_estimator=True, n_jobs=2)

def hyperparameter_tuning(model, grid, train_x, train_y, test_x, test_y, splits=10, repeats=10):
    '''
    Runs automatic hyperparameter tuning on classification models with 'model' and parameters specificed by 'grid'.
    Returns model with best results.
    '''
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv =cv, scoring="accuracy")
    gridResult = gridSearch.fit(train_x, train_y)
    
    print("Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))

    return gridResult.best_estimator_  