import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix,recall_score, classification_report, auc, roc_curve,roc_auc_score, fbeta_score

plt.style.use('seaborn')

def model_fit_with_grid_search_cv(model,parameters,X,y,folds = 5, score = 'accuracy',verbose = 0):
    start = time.time()
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        cv = StratifiedKFold(folds),
        scoring = score, 
        verbose = verbose,
        n_jobs = -1)
    grid_search.fit(X,y)

    if verbose > 0:
        print("--- Ellapsed time: %s seconds ---" % (time.time() - start))
        print('Best params: ',grid_search.best_params_)
        print('Best score (%s)' % score,grid_search.best_score_)
    return grid_search.best_estimator_,grid_search.best_params_, grid_search.best_score_

def model_fit_with_rfe_cv(model,X,y,folds = 5, score = 'f1',verbose = 0):
    
    rfecv = RFECV(
        estimator=model, 
        step=1, 
        cv=StratifiedKFold(folds),
        scoring=score,
        n_jobs = -1)
    rfecv.fit(X, y)
    
    selected_features = X.columns[rfecv.support_]
    highest_score = rfecv.grid_scores_.max();
    best_model = rfecv.estimator
    
    if verbose > 0:
        print("Optimal number of features : %d" % rfecv.n_features_)
        print('highest score: ', rfecv.grid_scores_.max())
    if verbose > 1:
        show_rfe_scores(rfecv);
    return selected_features, highest_score, best_model, rfecv

def show_rfe_scores(rfe):
    plt.figure()
    plt.title('RFE Scores')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()


def predict(model,X,y):
    df_result = pd.DataFrame(columns = ['TrueClass','Predicted'])
    df_result.Predicted = model.predict(X)
    df_result.TrueClass = y.values.ravel()
    return df_result
    

def evalute_model_performance(model,model_name,X,y):
    df_result = predict(model,X,y)
    class_report = classification_report(df_result.TrueClass, df_result.Predicted,output_dict = True)
    accuracy = class_report['accuracy']
    recall = class_report['macro avg']['recall']
    precision = class_report['macro avg']['precision']
    f1 = class_report['macro avg']['f1-score']
    f2 = fbeta_score(df_result.TrueClass, df_result.Predicted,beta = 2, average='macro')
    print('')
    print('Performance Report: ')
    print('Accuracy: %1.3f' % accuracy)
    print('Recall: %1.3f' % recall)
    print('Precision: %1.3f' % precision)
    print('F1: %1.3f' % f1)
    print('F2: %1.3f' % f2)
    print('')

    plot_confusion_matrix(df_result,model_name)

    try:
        plot_ROC(model, model_name, X, y)
    except:
        print('Could not print ROC AUC curve.')

def plot_confusion_matrix(df,title,labels = ['Negative', 'Positive'],dataset_type = 'Test'):
    conf_matrix = confusion_matrix(df.TrueClass, df.Predicted)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title('{0} - Confusion matrix - {1} set'.format(title,dataset_type), fontsize = 20)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    return conf_matrix.ravel()

def plot_ROC(model,model_name,X_test,y_test):
    
    naive_probs = [0 for _ in range(len(y_test))]
    
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]

    naive_auc = roc_auc_score(y_test, naive_probs)
    model_auc = roc_auc_score(y_test, probs)

    print('No Skill: ROC AUC=%.3f' % (naive_auc))
    print(model_name,': ROC AUC=%.3f' % (model_auc))

    naive_fpr, naive_tpr, _ = roc_curve(y_test, naive_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    
    plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive')
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    