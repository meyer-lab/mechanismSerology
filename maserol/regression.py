import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                            classification_report, roc_curve, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler

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

def get_crossval_info(model, train_x, train_y, splits=10, repeats=10):
    '''
    Crossvalidates regression using 'model'.
    '''
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    return cross_validate(model, train_x, train_y, cv=cv, return_estimator=True, n_jobs=2)

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

def plot_regression_roc_curve(models, test_x, test_y, colors):
    '''
    Plots ROC curve for each cross-validated model in 'models'. Number of elements in 'colors' must correspond to number
    of different types of regression models validated.
    '''
    text_y = 0.2
    for i in range(len(models)):
        roc_probs = []
        aucs = []
        for model in models[i]['estimator']:
            probs = model.predict_proba(test_x)[:, 1]
            roc_probs.append(probs) # keep probabilities for positive outcome only
            aucs.append(roc_auc_score(test_y, probs))

        for prob in roc_probs:
            lr_fpr, lr_tpr, _ = roc_curve(test_y, prob, pos_label='Severe')
            sns.lineplot(lr_fpr, lr_tpr, color=colors[i], ci=None, alpha=0.1)

        fpr, tpr, _ = roc_curve(test_y, np.mean(roc_probs, axis=0), pos_label='Severe') # average of all models 
        sns.lineplot(fpr, tpr, color=colors[i], ci=None, linewidth=3)
        plt.text(0.63, text_y,'AUC=%.3f' % (np.mean(aucs)) + ' \u00B1 %.2f' % np.std(aucs), color=colors[i])

        text_y -= 0.05
    
    f = sns.lineplot([x for x in range(0, 2)], [y for y in range(0,2)], linestyle="--", color="k")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    sns.despine(left=False, bottom=False)
    return f

def plot_regression_weights(cv_model, absf):
    '''
    Plots regression weighs for each component when using logistic regression.
    '''
    coefs = []
    for model in cv_model['estimator']:
        coefs.append(model.coef_)
    coefs = np.asarray(coefs)
    f = sns.barplot(data=np.reshape(coefs, (coefs.shape[0], coefs.shape[2])), palette='colorblind')
    f.set_xticklabels(absf)
    plt.axhline(0, color='k', clip_on=False, linestyle='--')
    plt.xlabel('Component')
    plt.ylabel('Component Weight')
    sns.despine(left=False, bottom=False)
    return f

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