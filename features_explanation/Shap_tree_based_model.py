# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:18:53 2021

@author: Cédric
"""

"""
In the medical field, being able to explain the outcome of a model is as
important as the result obtained with the model. Specialist need to understand
how the model came to this outcome to trust it.
"""
def prepare_explenation(rf, X, y_name):
    """
    In this part, we preprocess the data before extraction the features
    importance.

    Parameters
    ----------
    rf : object sklearn classifier (Tree model)
    X : DataFrame
    y_name : str

    Returns : DataFrame
    X_risk
    -------
    X_risk : DataFrame

    """
    
    X_risk = X.copy(deep=True)
    X_risk.loc[:, y_name] = rf.predict_proba(X_risk)[:,1]
    X_risk = X_risk.sort_values(by=y_name, ascending=False)
    
    return X_risk

def feature_importance(rf, X, y_name='risk', i):
    """
    Using this function, we can easily compare every feature importance for
    the final outcome.

    Parameters
    ----------
    rf : object sklearn classifier (Tree)
    X : DataFrame
    y_name : str, optionalThe default is 'risk'.
    i : int

    -------

    """
    
    explainer = shap.TreeExplainer(rf)
    i = i
    X_risk = prepare_explenation(rf, X, y_name)
    shap_value = explainer.shap_values(X_risk.loc[X_risk.index[i], :])[1]
    shap.force_plot(explainer.expected_value[1], shap_value,
                    feature_names=X_risk.columns, matplotlib=True)
    
def feature_use_summary(rf, X, y_name='risk'):
    """
    With the summary, we are able to visualize every predicted case,
    red color mean that the model predict for a condition and blue if
    the model predict for no conditions.

    Parameters
    ----------
    rf : object sklearn classifier (Tree)
    X : DataFrame
    y_name : str, optionalThe default is 'risk'.

    -------

    """
    
    explainer = shap.TreeExplainer(rf)
    X_risk = prepare_explenation(rf, X, y_name)
    shap_values = shap.TreeExplainer(rf).shap_values(X_risk)[1]
    shap.summary_plot(shap_values, X_risk)
    
def dependence_plot(rf, X, y_name='risk', col, interaction_index):
    """
    The dependence plot, plot every outcome on a graph. We can then vizualise
    the dependence between two features and the model outcome.
    
    Parameters
    ----------
    rf : object sklearn classifier (Tree)
    X : DataFrame
    y_name : str, optionalThe default is 'risk'.
    col : str
    interaction_index : str

    -------

    """
    
    explainer = shap.TreeExplainer(rf)
    X_risk = prepare_explenation(rf, X, y_name)
    shap_values = shap.TreeExplainer(rf).shap_values(X_risk)[1]
    
    shap.dependence_plot(col, shap_values, X_risk, interaction_index=interaction_index)