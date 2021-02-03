# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:33:46 2021

@author: Cédric
"""
import numpy as np
"""
We can improve models by adding interaction in between features.
"""

def add_interactions(X):
    """
    To add interaction, we multiply features by pairs row wise

    Parameters
    ----------
    X : DataFrame (pandas data structure)

    X_interaction : DataFrame X with n more features being the interaction
    -------

    """
    
    features = X.columns
    n = len(features)
    X_interaction = X.copy(deep=True)
    
    for i in range(n):
        feature_i_name = features[i]
        feature_i_data = X[feature_i_name]
        
        for j in range(i+1, n):
            feature_j_name = features[j]
            feature_j_data = X[feature_j_name]
            
            feature_i_j_name = feature_i_name+'_&_'+feature_j_name
            feature_i_j_data = feature_i_data * feature_j_data
            
            X_interaction[feature_i_j_name] = feature_i_j_data
            
    return X_interaction