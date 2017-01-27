import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sqlalchemy import create_engine

def load_data_from_database(connection_dd):

    """
    ............................................................
    Loads data using provided connection dictionary as argument
    
    This function uses sqlalchemy module.
    
    Output is a Pandas dataframe.
    ............................................................
    """   
    engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(connection_dd["user"], connection_dd["password"],
                                            connection_dd["url"], connection_dd["port"], connection_dd["database"]))
    df = pd.read_sql("SELECT * FROM {}".format(connection_dd["table"]), con=engine)

    return df

def make_data_dict(df, random_state=None):
    """
    ............................................................
    Creates data dictionary
    
    Takes a Pandas dataframe as an argument and seperates it by 
        columns into features matrix and target vector, then does 
        train, test split on them.
    
    This function uses sklearn.model_selection module.
    
    Output is a dictionary containing X_train, X_test, y_train, y_test,
        list of features(column names) and random state.
    ............................................................
    """   
    y_data = df['label']
    X_data = df.drop(['index','label'], axis=1)
    features = X_data.columns
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=random_state)
    return {
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test,
            'features' : features,
            'random_state' : random_state
        }

def general_transformer(dd, transformer, scale=False, random_state=None):
     
    """
    ............................................................
    Transforms data
    
    Takes a data dictionary, a transformer as arguments and does
        the transformation on data.
    - when the transformer is an scaling model the optional scale
        parameter should be True.
    
    Uses the train data to fit the transformer and then transforms
        the features of test and train.
    
    Output is a dictionary containing transformed X_train, 
        transformed X_test, y_train, y_test, corrected list of 
        features(column names) the transformer/the scaler and random state.
    ............................................................
    """   

    local_dd = dict(dd)
    
    if scale:
        transformer.fit(local_dd['X_train'])
        local_dd['scaler'] = transformer
    else:
        transformer.fit(local_dd['X_train'], local_dd['y_train'])
        local_dd['transformer'] = transformer
        local_dd["features"] = local_dd["features"][transformer.get_support()]
        
    
    local_dd['X_train'] = transformer.transform(local_dd['X_train'])
    local_dd['X_test'] = transformer.transform(local_dd['X_test'])
    
    return local_dd    

def general_model(dd, model, grid_search=False, random_state=None):
     
    """
    ............................................................
    Builds the model
    
    Takes a data dictionary and a model as arguments.
    
    - when the model is a grid search model the optional grid_search
        parameter should be True.
    
    Uses the train data to fit the model and then finds the relative 
        score for test and train data.
    
    Output is a dictionary containing the values of input dictionary 
        plus test scores for train and test data, model(in case of 
        grid search returns the grid search model and best estimator)
        and the random state.
        
        - if the model used is Logistic Regression, coefficients will
        be added to the features at the output dictionary.
        
        - if inside a grid search a KNN method is used, number of 
        neighbors corresponding to the best result will be included in
        output dictionary.
    ............................................................
    """   

    local_dd = dict(dd)    
    model.fit(local_dd['X_train'], local_dd['y_train'])
    
    if grid_search:
        local_dd['model'] = model.best_estimator_
        local_dd['grid_search'] = model
        if str(local_dd['model'])[0:20] == "KNeighborsClassifier":
            local_dd["best_n_neighbors"] = local_dd['model'].n_neighbors
    else:
        local_dd['model'] = model

    local_dd['train_score'] = local_dd['model'].score(local_dd['X_train'], local_dd['y_train'])
    local_dd['test_score'] = local_dd['model'].score(local_dd['X_test'], local_dd['y_test'])

    if str(local_dd['model'])[0:18] == "LogisticRegression":
        coefs = local_dd['model'].coef_[0]
        local_dd["features"] = zip(local_dd["features"],coefs)
            
    return local_dd    

def skb_gridsearch_results(l_dd, fs_params):
    """
    ............................................................
    Finds the best results for manual grid search
    
    Takes a list of data dictionaries and a range of values as arguments.

    Uses the list of data dictionaries which are outputs of running
        a model for different k values for KBest transformer, and finds
        the k value, the data and the model which produced the best results.
        
    Output is one of the dictionaries of the input list plus the value
        of the k which has produced the best results.
    ............................................................
    """   

    k_range = fs_params
    k_range_min = min(k_range)
    k_range_max = max(k_range)
    max_test_score = 0
    for i in range(k_range_max - k_range_min+1):
        tst_s = l_dd[i]["test_score"]
        if max_test_score <= tst_s:
            max_test_score = tst_s
            max_k = i + k_range_min
            ind_k_best = i
    dd = dict(l_dd[ind_k_best])
    dd["best_k"] = max_k
    
    return dd


def pipeline(connection_dd, scaler, transformer=None, model=None, fs_params=None, gs_params=None, verbose=True, random_state=None):
    """
    ............................................................
    A full pipeline for reading data from a remote database and 
        finding the best model
        
    - use verbose=False to avoid receiving messages
    
    ............................................................
    """   

    data = load_data_from_database(connection_dd)
    if verbose:
        print "Connected to the database and got the data successfully."
    dd1 = make_data_dict(data, random_state=random_state)
    if verbose:
        print "Data dictionary created."
    dd2 = general_transformer(dd1, scaler, scale=True, random_state=random_state)
    if verbose:
        print "Data is scaled."
    
    if transformer:
        if verbose:
            print "Transformer is found."
        if fs_params:
            if verbose:
                print "Transformer parameters are found."
            l_dd4 = []
            if gs_params:
                if verbose:
                    print "Grid searches are created."
            else:
                if verbose:
                    print "No grid search."
            for ks in fs_params:
                dd3 = general_transformer(dd2, transformer(k=ks), random_state=random_state)
                if gs_params:
                    gs = GridSearchCV(model, param_grid=gs_params)
                    dd4 = general_model(dd3, gs, grid_search=True)
                    l_dd4.append(dd4)
                else:
                    l_dd4.append(general_model(dd3, model))
            if gs_params:
                if verbose:
                    print "Grid searches are done."
            dd4 = skb_gridsearch_results(l_dd4, fs_params)
            return dd4
        else:
            if verbose:
                print "Transformer parameters are not found, using default."
            dd3 = general_transformer(dd2, transformer, random_state=random_state)
            if gs_params:
                gs = GridSearchCV(model, param_grid=gs_params)
                if verbose:
                    print "Grid search is created."
                dd4 = general_model(dd3, gs, grid_search=True)
                if verbose:
                    print "Grid search is done."
            else:
                if verbose:
                    print "No grid search."
                dd4 = general_model(dd3, model)                
            dd4["best_k"] = "Default value of k=10"
            return dd4
    else:
        if verbose:
            print "Transformer is  not found."
        if gs_params:
            gs = GridSearchCV(model, param_grid=gs_params)
            if verbose:
                print "Grid search is created"
            dd4 = general_model(dd2, gs)
            if verbose:
                print "Grid search is done."
        else:
            if verbose:
                print "No grid search."
            dd4 = general_model(dd2, model)                            
        dd4["best_k"] = None
        return dd4
 