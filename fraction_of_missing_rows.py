# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:03:35 2021

@author: Cédric
"""
import pandas as pd
import numpy as np

def fraction_of_missing_rows(df):
    """
    pourcent of rows with at least one missing feature

    Parameters
    ----------
    df : DataFrame

    float (0.0-1.0)
    -------

    """
    return np.sum(pd.DataFrame.any(df.isnull(),axis=1)) / len(df)