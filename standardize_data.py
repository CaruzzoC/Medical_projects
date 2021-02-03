# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:11:37 2021

@author: Cédric
"""
import numpy as np
"""
Standardizing is a key for any use of data through Machine learning and
Deep learning. In order to justify this, we can take the example of
gradient descent and parameters topology.The parameters with the greatest
range dominate the loss update. Therefore, we put every parameter on
a same scale to avoid this issue.
"""

def standardize_data(df_train, df_test):
    """
    To make the data closer to a normal distribution (and so
    to use the normalization function) we firstly take the log of it.
    And then we apply the function of standardization : x' = (x - mean) / std
    We will use the mean and std of the df_train not to bias the data.

    Parameters
    ----------
    df_train : DataFrame (pandas data structure)
    df_test : DataFrame (pandas data structure)

    return : df_train and df_test standardized
    DataFrame, DataFrame (pandas data structure)
    df_train_standardized, df_test_standardized
    -------

    """
    
    #Step 1 making the data closer to a normal distribution
    df_train_log = np.log(df_train)
    df_test_log = np.log(df_test)
    
    #Step 2 getting the mean and the std (training set)
    mean = df_train_log.mean(axis=0)
    std = df_train_log.std(axis=0)
    
    #Step 3 standardize the data
    df_train_standardized = (df_train - mean) / std
    df_test_standardized = (df_test - mean) / std
    
    return df_train_standardized, df_test_standardized