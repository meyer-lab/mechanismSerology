import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


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