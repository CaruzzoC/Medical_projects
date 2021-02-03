# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:48:22 2021

@author: Cédric
"""

"""
For medical prognosis, one of the main metric to evaluate a model is
the C-index.
The C-index measures the discriminatory power of a risk score.
Intuitively, the goal is to maximize the score.
the formula is the following : (concordants + 0.5 * ties) / permissibles
"""
import numpy as np

def C_index(y_true, scores):
    """
    Firstly, we have to count every permissible pairs, concordant pairs
    and ties.
    To be permissible each individu of the pair must admit different outcomes
    To be concordant the individu with the condition must admit a higher score
    To be a tie, both individu must admit the same score.

    Parameters
    ----------
    y_true : 1-D np.array
        binary array, 0 : patient does not get the condition;
                      1 : patient does get the codition.
    scores :1-D np.array
        Corresponding risk scores.

    C_index_score (float) 
    result of the C_index formula : (concordants + 0.5 * ties) / permissibles
    -------

    """
    
    n = len(y_true)
    assert len(scores) == n
    
    concordant = 0
    permissible = 0
    ties = 0
    
    for i in range(n):
        for j in range(i+1, n):
            
            if y_true[i] != y_true[j]:
                permissible += 1
                
                if scores[i] == scores[j]:
                    ties += 1
                
                elif y_true[i] == 0 and y_true[j] == 1:
                    
                    if scores[i] < scores[j]:
                        concordant += 1
                        
    c_index = (concordant + 0.5 * ties) / permissible 
    
    return c_index
                