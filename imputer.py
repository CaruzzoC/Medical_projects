# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 22:47:14 2021

@author: Cédric
"""

"""
In the medical field, it is not rare to miss a lot of data. To counter this
issue, we have imputation techniques. Mainly : mean, regression, drop
"""

def simple_impute_data(strategy = 'mean', X_train, X_val):
    """
    We use the SimpleImputer to impute the data

    Parameters
    ----------
    strategy : str, optional
        strategy to use when imputing. The default is 'mean'.
    X_train : DataFrame
    X_val : DataFrame

    Returns : DataFrame, DataFrame
    X_train_imputed, X_val_imputed
    -------
    X_train_imputed : DataFrame
    X_val_imputed : DataFrame

    """
    
    imputer = SimpleImputer(strategy = strategy)
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train),
                                        columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val),
                                 columns=X_val.columns)
    
    return X_train_imputed, X_val_imputed

def regression_impute_data(random_state=0, sample_posterior=False, max_iter=1, min_value=0, X_train, X_valid):
    """
    The regression impute can be the best option if the data admit a linear 
    correlation

    Parameters
    ----------
    random_state : int, optional
        The default is 0.
    sample_posterior : bool, optional
        The default is False.
    max_iter : int, optional
        The default is 1.
    min_value : int, optional
        The default is 0.
    X_train : DataFrame
    X_valid : DataFrame

    Returns : DataFrame, DataFrame
    X_train_imputed, X_val_imputed
    -------

    """
    imputer = IterativeImputer(random_state=random_state,
                               sample_posterior=sample_posterior,
                               max_iter = max_iter,
                               min_value=min_value)
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train),
                                   columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val),
                                 columns=X_val.columns)
    
    return X_train_imputed, X_val_imputed