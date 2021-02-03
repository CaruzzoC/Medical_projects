# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:08:44 2021

@author: Cédric
"""

"""
Hyperparameters tunning is an empirical process. We then can use automation to
do it for us. I will implement here the regular grid search. There is a way to
accelerate the process, which is the random grid search.
"""
def grid_search(clf, X_train, y_train, X_val, y_val, hyperparams, fixed_hyperparams={}):
    """
    We will use the combination of every hyperparams and evaluate the model
    with the C-index. The best combination is then returned

    Parameters
    ----------
    clf : object sklearn classifier
    X_train : DataFrame
    y_train : DataFrame
    X_val : DataFrame
    y_val : DataFrame
    hyperparams : Dict
    fixed_hyperparams : Dict, optional
        The default is {}.

    Returns Best_clf : object sklearn classifier
            Best_hyperparams : dict
    -------

    """
    
    best_estimator = None
    best_hyperparams = {}
    
    best_score = 0
    
    lists = hyperparams.values()
    
    param_combi = list(itertools.product(*lists))
    total_param_combi = len(param_combi)
    
    for i, params in enumerate(param_combi, 1):
        param_dict = {}
        
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        estimator = clf(**param_dict, **fixed_hyperparams)
        
        estimator.fit(X_train, y_train)
        
        preds = estimator.predict_proba(X_val)
        
        estimator_score = C_index(y_val, preds[:,1])
        
        print(f'[{i}/{total_param_combi}] {param_dict}')
        print(f'Val C-Index: {estimator_score}\n')
        
        if estimator_score > best_score:
            best_score = estimator_score
            best_hyperparams = param_dict
            
    best_hyperparams.update(fixed_hyperparams)
    
    return best_estimator, best_hyperparams


"""
Application of the previous function
"""
def clf_grid_search(clf, X_train, y_train, X_val, y_val, hyperparams_in, fixed_hyperparams_in={}):
    """
    

    Parameters
    ----------
    clf : object sklearn classifier
    X_train : DataFrame
    y_train : DataFrame
    X_val : DataFrame
    y_val : DataFrame
    hyperparams : dict
    fixed_hyperparams : dict, optional
       The default is {}.

    Returns
    -------

    """
    
    hyperparams = hyperparams_in
    fixed_hyperparams = fixed_hyperparams_in
    
    clf = clf
    
    best_clf, best_hyperparams = grid_search(clf, X_train, y_train,
                                                    X_val, y_val, hyperparams,
                                                    fixed_hyperparams)
    
    print(f"Best hyperparameters:\n{best_hyperparams}")

    
    y_train_best = best_clf.predict_proba(X_train)[:, 1]
    print(f"Train C-Index: {C_index(y_train, y_train)}")

    y_val_best = best_clf.predict_proba(X_val)[:, 1]
    print(f"Val C-Index: {C_index(y_val, y_val_best)}")
    
    best_hyperparams.update(fixed_hyperparams)
    
    return best_clf, best_hyperparams
        