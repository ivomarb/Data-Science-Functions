# Import all global variables and basic libraries
import pandas as pd
import matplotlib.pyplot as plt

#############################################################################################################
# CLASSIFICATION METHODS
#############################################################################################################

def fit_predict_plot(X_train, X_test, y_train, y_test, models, print_only_table=False):
    """
    Fits the l_model using the X_train and y_train datasets. Accuracy on the train and test sets.
    Plots confusion matrix and classification report. Adds to an output dataset all the scores. 
    (Precision, Recall, F1-Score, Support for both classes: 0 and 1.)
    
    Args:
        X_train, X_test, y_train, y_test: The train and tests datasets.
        models: A double list with the classifier name as string and classifier instance.

    Returns:
        dataset: Returns output scores dataset.
    """
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test,  pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.Series)
    assert isinstance(y_test,  pd.core.frame.Series)
    assert isinstance(models,  (list,tuple))

    import sklearn.metrics

    # The *a syntax unpacks the multidimensional array into single array arguments.
    models_zip = list(zip(*models))
    
    output_scores_dataset = pd.DataFrame(index = ['Precision 0', 'Recall 0', 'F1-Score 0' , 'Support 0', 
                                                  'Precision 1', 'Recall 1', 'F1-Score 1' , 'Support 1'] , 
                                         columns = models_zip[0])

    for name, model in models:
        if print_only_table is False:
            print('------------------------------------------------------------------------------')
            print(name)
            print('------------------------------------------------------------------------------')

        #Fitting the l_model.
        model.fit(X_train, y_train)

        #Measuring accuracy.
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        accuracy_train = sklearn.metrics.accuracy_score(y_train, y_train_pred)
        accuracy_test = sklearn.metrics.accuracy_score(y_test, y_test_pred)

        if print_only_table is False:
            print('Accuracy on the train set: {}'.format(accuracy_train))
            print('Accuracy on the test set:  {}'.format(accuracy_test))

        #Plotting confusion matrix.
        cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_test_pred)
        true_negative, false_positive, false_negative, true_positive = \
                                                           sklearn.metrics.confusion_matrix(y_test, y_test_pred).ravel()
        
        if print_only_table is False:
            plt.figure()
            # Implemented at plotting.py
            plot_confusion_matrix(cnf_matrix, ['Negative', 'Positive'], title='Confusion matrix')
            plt.show()

        #Showing classification report.
        class_report = sklearn.metrics.classification_report(y_test, y_test_pred)
        if print_only_table is False:
            print(class_report)

        # Printing scores to output dataset.
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(y_test, y_test_pred)

        output_scores_dataset.loc['Precision 0',              name] = float("{0:.2f}".format(precision[0]))
        output_scores_dataset.loc['Recall 0',                 name] = float("{0:.2f}".format(recall[0]))
        output_scores_dataset.loc['F1-Score 0',               name] = float("{0:.2f}".format(fscore[0]))
        output_scores_dataset.loc['Support 0',                name] = float("{0:.2f}".format(support[0]))
        output_scores_dataset.loc['Precision 1',              name] = float("{0:.2f}".format(precision[1]))
        output_scores_dataset.loc['Recall 1',                 name] = float("{0:.2f}".format(recall[1]))
        output_scores_dataset.loc['F1-Score 1',               name] = float("{0:.2f}".format(fscore[1]))
        output_scores_dataset.loc['Support 1',                name] = float("{0:.2f}".format(support[1]))
        output_scores_dataset.loc['True Positive',            name] = true_positive
        output_scores_dataset.loc['False Positive',           name] = false_positive
        output_scores_dataset.loc['True Negative',            name] = true_negative
        output_scores_dataset.loc['False Negative',           name] = false_negative
        output_scores_dataset.loc['Accuracy on Training Set', name] = float("{0:.2f}".format(accuracy_train))
        output_scores_dataset.loc['Accuracy on Test Set',     name] = float("{0:.2f}".format(accuracy_test))
    
    # Can use idxmax with axis=1 to find the column with the greatest value on each row.
    output_scores_dataset['Max Value']      = output_scores_dataset.apply(max, axis = 1)
    #output_scores_dataset['Max Classifier'] = output_scores_dataset.idxmax(axis=1)

    return output_scores_dataset

def run_one_classifier(model, X_train, X_test, y_train, y_test):
    """
    Run a classifier model in the dataset and print results
    Args:
        model: Sklearn classifier model
        a_X_train, a_X_test, a_y_train, a_y_test: The train and tests datasets.
    """

    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test,  pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.Series)
    assert isinstance(y_test,  pd.core.frame.Series)


    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import accuracy_score, classification_report
    import seaborn as sns
    
    model.fit(X_train,y_train)
    ac = accuracy_score(y_test,model.predict(X_test))
    print('Accuracy is: ',ac)
    print(classification_report(y_test, model.predict(X_test)))
    cm = confusion_matrix(y_test,model.predict(X_test))
    sns.heatmap(cm,annot=True,fmt="d")
    return model

def run_all_classifiers(X_train, X_test, y_train, y_test, print_output_scores_to_csv=False, output_scores_csv_file_suffix='', print_only_table=False):
    """
    The list of all classifiers was generated by running the following commented code.

    Args:
        a_X_train, a_X_test, a_y_train, a_y_test: The train and tests datasets.
        a_print_output_scores_to_csv: If True the Precision, Recall, F1-Score and Support for both classes will
        be printed to a file with the current date and time.
        a_output_scores_csv_file_suffix: Suffix to be added to the csv file just before the .csv extension. Normally
        describing the run that is being performed.

    Returns:
        dataset: Returns output scores dataset.

    """
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test,  pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.Series)
    assert isinstance(y_test,  pd.core.frame.Series)
    assert isinstance(print_output_scores_to_csv, bool)
    assert isinstance(output_scores_csv_file_suffix, object)

    import time

    # https://stackoverflow.com/questions/42160313/how-to-list-all-classification-regression-clustering-algorithms-in-scikit-learn
    #from sklearn.utils.testing import all_estimators
    #estimators = all_estimators()
    #for name, class_ in estimators:
    #    log_print(name)

    from sklearn.calibration           import CalibratedClassifierCV
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble              import AdaBoostClassifier
    from sklearn.ensemble              import BaggingClassifier
    from sklearn.ensemble              import ExtraTreesClassifier
    from sklearn.ensemble              import GradientBoostingClassifier
    from sklearn.ensemble              import RandomForestClassifier
    from sklearn.gaussian_process      import GaussianProcessClassifier
    from sklearn.linear_model          import LogisticRegression
    from sklearn.linear_model          import LogisticRegressionCV
    from sklearn.linear_model          import SGDClassifier

    from sklearn.mixture               import BayesianGaussianMixture
    from sklearn.mixture               import DPGMM
    from sklearn.mixture               import GaussianMixture
    from sklearn.mixture               import GMM
    from sklearn.mixture               import VBGMM
    from sklearn.naive_bayes           import BernoulliNB
    from sklearn.naive_bayes           import GaussianNB
    from sklearn.neighbors             import KNeighborsClassifier
    from sklearn.neural_network        import MLPClassifier
    from sklearn.semi_supervised       import LabelPropagation
    from sklearn.semi_supervised       import LabelSpreading
    from sklearn.svm                   import SVC
    from sklearn.tree                  import DecisionTreeClassifier
    #from xgboost                       import XGBClassifier

    models = []
    models.append(('AdaBoostClassifier',            AdaBoostClassifier()))
    models.append(('BaggingClassifier',             BaggingClassifier()))
    models.append(('BayesianGaussianMixture',       BayesianGaussianMixture()))
    models.append(('BernoulliNB',                   BernoulliNB()))
    models.append(('CalibratedClassifierCV',        CalibratedClassifierCV()))
    models.append(('DPGMM',                         DPGMM()))
    models.append(('DecisionTreeClassifier',        DecisionTreeClassifier(random_state=0)))
    models.append(('ExtraTreesClassifier',          ExtraTreesClassifier(random_state=0)))
    models.append(('GMM',                           GMM()))
    models.append(('GaussianMixture',               GaussianMixture()))
    models.append(('GaussianNB',                    GaussianNB()))
    models.append(('GaussianProcessClassifier',     GaussianProcessClassifier()))
    models.append(('GradientBoostingClassifier',    GradientBoostingClassifier()))
    models.append(('KNeighborsClassifier',          KNeighborsClassifier()))
    models.append(('LabelPropagation',              LabelPropagation()))
    models.append(('LabelSpreading',                LabelSpreading()))
    models.append(('LinearDiscriminantAnalysis',    LinearDiscriminantAnalysis()))
    models.append(('LogisticRegression',            LogisticRegression()))
    models.append(('LogisticRegressionCV',          LogisticRegressionCV()))
    models.append(('MLPClassifier',                 MLPClassifier()))
    #models.append(('MultinomialNB', MultinomialNB()))
    #models.append(('NuSVC', NuSVC()))
    models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))
    models.append(('RandomForestClassifier',        RandomForestClassifier(random_state=g_seed)))
    models.append(('SGDClassifier',                 SGDClassifier()))
    models.append(('SVC',                           SVC()))
    models.append(('VBGMM',                         VBGMM()))
    #models.append(('XGBClassifier',                 XGBClassifier()))
    
    output_scores_df = fit_predict_plot(X_train, X_test, y_train, y_test, models, print_only_table)

    if print_output_scores_to_csv:
        output_scores_df.to_csv(time.strftime('output_scores' + str(output_scores_csv_file_suffix) + '.csv')

    return output_scores_df

def train_test_split_for_classification(dataset, test_size, random_state):
    """
    Selects X and y, considering that y has been renamed to label.
    """
    from sklearn.model_selection import train_test_split

    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(test_size, float)
    assert isinstance(random_state, int)

    X = dataset.loc[:, dataset.columns != g_label]
    y = dataset[g_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    log_print('X_train: {}'.format(X_train.shape))
    log_print('y_train: {}'.format(y_train.shape))
    log_print('X_test:  {}'.format(X_test.shape))
    log_print('y_test:  {}'.format(y_test.shape))
    return(X_train, X_test, y_train, y_test)