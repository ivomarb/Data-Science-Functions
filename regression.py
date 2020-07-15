# Import all global variables and basic libraries
import numpy as np
import pandas as pd

SEED = 0

#############################################################################################################
# REGRESSION METHODS
#############################################################################################################

def mean_absolute_percentage_error(y, y_pred):
    """
    Compute the mean absolute percentage error. It usually expresses accuracy as a percentage.

    Args:
        y (list): list of real values of target
        y_pred (list): list of predicted values for target

    Returns:
        float: mean absolute percentage error
    """

    y = np.array(y)
    y_pred = np.array(y_pred)[y != 0]
    y = y[y != 0]
    return np.mean(np.abs((y - y_pred) / y)) * 100

def print_results(y, y_pred):
    """
    Print mean squared error, mean absolute error, r2 score.

    Args:
        y (list): list of real values of target
        y_pred (list): list of predicted values for target
    """

    assert isinstance(y, pd.core.series.Series)
    assert isinstance(y_pred, np.ndarray)

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    mean_squared_error = np.sqrt(mean_squared_error(y, y_pred))
    mean_absolute_percentage_error_value = mean_absolute_percentage_error(y, y_pred)
    r2_score = r2_score(y, y_pred)

    print("mean squared error: "+    str(mean_squared_error))
    print("mean absolute % error: "+ str(mean_absolute_percentage_error_value))
    print("r2 score: "+ str(r2_score))


def save_best_regression_model(X_train, y_train):
    """
    Create and save the regression model that obtained best results in several tests

    Args:
        X_train (matrix): matrix with features
        y_train (list): list of values of target
    """

    from sklearn import ensemble
    from acq_util import util_save_model_pkl

    best_params = {'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    model_name = 'randomforestregressor'

    model = ensemble.RandomForestRegressor(random_state=SEED, **best_params)
    model = train_regressor(model, model_name, X_train, y_train)
    util_save_model_pkl(model, model_name+'.pkl')

def train_regressor(model, model_name, X_train, y_train):
    """
    Train a regression model

    Args:
        model: sklearn regression model
        model_name (string): name of the model
        X_train (matrix): matrix with features
        y_train (list): list of values of target
    Returns:
        sklearn model: trained regression model
    """
    import sklearn.base

    assert isinstance(model, sklearn.base.RegressorMixin)
    assert isinstance(model_name, object)
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.series.Series)

    print('------------------------------------------------------------------------------')
    print(model_name)
    print('------------------------------------------------------------------------------')

    model.fit(X_train, y_train)

    return model

def train_test_regressor(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and test a regression model and print the results

    Args:
        model: sklearn regression model
        model_name (string): name of the model
        X_train (matrix): matrix with features of the training set
        y_train (list): list of values of target of the training set
        X_test (matrix): matrix with features of the test set
        y_test (list): list of values of target of the test set
    """
    import sklearn.base

    assert isinstance(model,       sklearn.base.RegressorMixin)
    assert isinstance(model_name,  object)
    assert isinstance(X_train,    (np.ndarray, np.generic, pd.core.frame.DataFrame))
    assert isinstance(X_test,     (np.ndarray, np.generic, pd.core.frame.DataFrame))
    assert isinstance(y_train,    (np.ndarray, np.generic,pd.core.series.Series))
    assert isinstance(y_test,     (np.ndarray, np.generic,pd.core.series.Series))

    print('------------------------------------------------------------------------------')
    print(model_name)
    print('------------------------------------------------------------------------------')

    model.fit(X_train, y_train)

    print('Training Set')
    y_pred = model.predict(X_train)
    print_results(y_train, y_pred)

    print('Testing Set')
    y_pred = model.predict(X_test)
    print_results(y_test, y_pred)

def train_test_all_regressors(X_train, X_test, y_train, y_test, seed=SEED):
    """
    Train, test and print the results of most available regressors presented in sklearn.

    Args:
        X_train (matrix): matrix with features of the training set
        y_train (list): list of values of target of the training set
        X_test (matrix): matrix with features of the test set
        y_test (list): list of values of target of the test set
    """
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.series.Series)
    assert isinstance(y_test, pd.core.series.Series)
    assert isinstance(seed, int)

    from sklearn import linear_model
    from sklearn import tree
    from sklearn import ensemble
    from sklearn import neighbors
    from sklearn import neural_network

    models = []
    models.append(("BayesianRidge",              linear_model.BayesianRidge()))
    models.append(("ElasticNet",                 linear_model.ElasticNet()))
    models.append(("HuberRegressor",             linear_model.HuberRegressor()))
    models.append(("Lars",                       linear_model.Lars()))
    models.append(("Lasso",                      linear_model.Lasso()))
    models.append(("LassoLars",                  linear_model.LassoLars()))
    models.append(("LinearRegression",           linear_model.LinearRegression()))
    models.append(("OrthogonalMatchingPursuit",  linear_model.OrthogonalMatchingPursuit()))
    models.append(("PassiveAggressiveRegressor", linear_model.PassiveAggressiveRegressor()))
    models.append(("Ridge",                      linear_model.Ridge()))
    models.append(("SGDRegressor",               linear_model.SGDRegressor()))
    models.append(("AdaBoostRegressor",          ensemble.AdaBoostRegressor(random_state=seed)))
    models.append(("BaggingRegressor",           ensemble.BaggingRegressor(random_state=seed)))
    models.append(("ExtraTreesRegressor",        ensemble.ExtraTreesRegressor(random_state=seed)))
    models.append(("GradientBoostingRegressor",  ensemble.GradientBoostingRegressor(random_state=seed)))
    models.append(("RandomForestRegressor",      ensemble.RandomForestRegressor(random_state=seed)))
    models.append(("DecisionTreeRegressor",      tree.DecisionTreeRegressor(random_state=seed)))
    models.append(("KNeighborsRegressor",        neighbors.KNeighborsRegressor()))
    models.append(("MLPRegressor",               neural_network.MLPRegressor()))

    best_mean_absolute_percentage_error = 100
    best_model = ''

    for name, model in models:
        print('------------------------------------------------------------------------------')
        print(name)
        print('------------------------------------------------------------------------------')

        model.fit(X_train, y_train)

        print('Training Set')
        y_pred = model.predict(X_train)
        print_results(y_train, y_pred)

        print('Testing Set')
        y_pred = model.predict(X_test)
        print_results(y_test, y_pred)

        mean_absolute_percentage_error_value = mean_absolute_percentage_error(y_test, y_pred)
        if  mean_absolute_percentage_error_value < best_mean_absolute_percentage_error:
            best_mean_absolute_percentage_error = mean_absolute_percentage_error
            best_model = name

    print('------------------------------------------------------------------------------')
    print('Best model: ' + best_model)
    print('Best mean absolute percentage error: ' + str(best_mean_absolute_percentage_error))
    print('------------------------------------------------------------------------------')

def train_test_all_regressors_with_cross_validation(X, y, seed=SEED):
    """
    Train, test and print the results of most available regressors presented in sklearn using cross validation.
    Args:
        X_train (matrix): matrix with features of the training set
        y_train (list): list of values of target of the training set
        X_test (matrix): matrix with features of the test set
        y_test (list): list of values of target of the test set
    """
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(y, pd.core.series.Series)
    assert isinstance(seed, int)
    
    from sklearn import linear_model
    from sklearn import tree
    from sklearn import ensemble
    from sklearn import neighbors
    from sklearn import neural_network
    
    from sklearn.model_selection import cross_val_score

    models = []
    models.append(("BayesianRidge",              linear_model.BayesianRidge()))
    models.append(("ElasticNet",                 linear_model.ElasticNet()))
    models.append(("HuberRegressor",             linear_model.HuberRegressor()))
    models.append(("Lars",                       linear_model.Lars()))
    models.append(("Lasso",                      linear_model.Lasso()))
    models.append(("LassoLars",                  linear_model.LassoLars()))
    models.append(("LinearRegression",           linear_model.LinearRegression()))
    models.append(("OrthogonalMatchingPursuit",  linear_model.OrthogonalMatchingPursuit()))
    models.append(("PassiveAggressiveRegressor", linear_model.PassiveAggressiveRegressor()))
    models.append(("Ridge",                      linear_model.Ridge()))
    models.append(("SGDRegressor",               linear_model.SGDRegressor()))
    models.append(("AdaBoostRegressor",          ensemble.AdaBoostRegressor(random_state=seed)))
    models.append(("BaggingRegressor",           ensemble.BaggingRegressor(random_state=seed)))
    models.append(("ExtraTreesRegressor",        ensemble.ExtraTreesRegressor(random_state=seed)))
    models.append(("GradientBoostingRegressor",  ensemble.GradientBoostingRegressor(random_state=seed)))
    models.append(("RandomForestRegressor",      ensemble.RandomForestRegressor(random_state=seed)))
    models.append(("DecisionTreeRegressor",      tree.DecisionTreeRegressor(random_state=seed)))
    models.append(("KNeighborsRegressor",        neighbors.KNeighborsRegressor()))
    models.append(("MLPRegressor",               neural_network.MLPRegressor()))

    best_rmse = 1000000000.0
    best_model = ''

    for name, model in models:
        print('------------------------------------------------------------------------------')
        print(name)
        print('------------------------------------------------------------------------------')

        scores = cross_val_score(model, X, y, scoring = 'neg_root_mean_squared_error', cv=5)
        scores = -scores
        scores_mean = scores.mean()
        scores_std = scores.std()
        print("RMSE: %0.3f (+/- %0.2f)" % (scores_mean, scores_std * 2))
        
        #mean_absolute_percentage_error_value = mean_absolute_percentage_error(y_test, y_pred)
        if  scores_mean < best_rmse:
            best_rmse = scores_mean
            best_model = name

    print('------------------------------------------------------------------------------')
    print('Best model: ' + best_model)
    print('Best RMSE: ' + str(best_rmse))
    print('------------------------------------------------------------------------------')

def train_test_split_for_regression_dataframe(dataset, test_size=0.2, random_state=SEED):
    """
    Split the dataset into train and test sets

    Args:
        dataset (DataFrame): dataset with all features; 
    Returns:
        X_train (matrix): matrix with features of the training set
        y_train (list): list of values of target of the training set
        X_test (matrix): matrix with features of the test set
        y_test (list): list of values of target of the test set
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(test_size, float)
    assert isinstance(random_state, int)

    from sklearn.model_selection import train_test_split
    from libs_global import g_listing_info__list_price
    
    X = dataset.loc[:, dataset.columns != g_listing_info__list_price]
    y = dataset[g_listing_info__list_price]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('X_train: '+ str(X_train.shape))
    print('y_train: '+ str(y_train.shape))
    print('X_test: '+  str(X_test.shape))
    print('y_test: '+  str(y_test.shape))
    return(X_train, X_test, y_train, y_test)

def train_test_split_for_regression(X, y, test_size=0.2, random_state=SEED):
    """
    Split the dataset into train and test sets

    Args:
        X (matrix): dataset with all features; 
        y (vector): the target
    Returns:
        X_train (matrix): matrix with features of the training set
        y_train (list): list of values of target of the training set
        X_test (matrix): matrix with features of the test set
        y_test (list): list of values of target of the test set
    """
    assert isinstance(test_size, float)
    assert isinstance(random_state, int)

    from sklearn.model_selection import train_test_split
      
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('X_train: '+ str(X_train.shape))
    print('y_train: '+ str(y_train.shape))
    print('X_test: ' + str(X_test.shape))
    print('y_test: ' + str(y_test.shape))
    return(X_train, X_test, y_train, y_test)
