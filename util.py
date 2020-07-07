def equalize_classes_by_frac_of_minority_class(X, y, label, frac=1.0):
    """
    Equalize classes by fraction of minority class.
    """
    import pandas as pd
    import numpy as np
    
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(y, pd.core.frame.DataFrame)
    assert isinstance(frac, float)

    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    num_min = np.min([num_neg, num_pos])
    num_max = np.max([num_neg, num_pos])

    dataset_full = X
    dataset_full[label] = y

    frac_to_remove = 1 - frac * num_min / num_max

    if num_neg > num_pos:
        dataset_full.drop(dataset_full.query(label == 0).sample(a_frac=frac_to_remove).index, inplace=True)
    else:
        dataset_full.drop(dataset_full.query(label == 1).sample(a_frac=frac_to_remove).index, inplace=True)

    
    y = dataset_full[label]
    X = dataset_full.loc[:, dataset_full.columns != label]

    return (X, y)

def get_numerical_columns_names(dataset):
    """
    Get the names of columns with numerical features

    Args:
        dataset (DataFrame): data

    Returns:
        list: list with names of numerical columns
    """
    import pandas as pd
    
    assert isinstance(dataset, pd.core.frame.DataFrame)

    numerical_features = dataset.select_dtypes(include=["int64", "float64"]).columns
    return numerical_features

def missing_values_report(df_):
    """
    Return a dataframe with the count of missing values and the ratio.
    """
    import pandas as pd
    
    count_missing = df_.isnull().sum().values
    ratio_missing = count_missing / df_.shape[0]

    return pd.DataFrame(data = {'count_missing': count_missing,
                                'ratio_missing': ratio_missing},
                        index = df_.columns.values).sort_values(by='ratio_missing',ascending=False)

def numeric_exploratory_data_analysis(dataset, generate_csv=False, print_statistical_summary = False, generate_unique_values = True, dataset_name = ''):
    """
    Outputs a pandas dataframe with the dtypes, nan and zero value counts of all features. 
    Also outputs a csv file.
    """
    import pandas as pd
    import numpy as np
    
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(generate_csv, bool)
        
     # Shape of the a_dataset.
    print('Shape: {} \n'.format(dataset.shape))

    # Summary and statistical description of the a_dataset.
    if print_statistical_summary:
      print('Statistical Summary: \n {} \n'.format(dataset.describe(include="all")))
    features = list(dataset)

    numeric_analysis = pd.DataFrame(index=features, columns=['Dtypes',
                                                             'N_of_non_NaN_values',
                                                             'N_of_NaN_values',
                                                             'N_of_non_zero_values',
                                                             'N_of_zero_values',
                                                             'N_of_unique_values',
                                                             'Unique_values'])

    numeric_analysis['Dtypes']               = dataset.dtypes
    numeric_analysis['N_of_non_NaN_values']  = dataset.count()
    numeric_analysis['N_of_NaN_values']      = len(dataset) - dataset.count()
    numeric_analysis['N_of_non_zero_values'] = dataset.astype(bool).sum(axis=0)
    numeric_analysis['N_of_zero_values']     = len(dataset) - dataset.astype(bool).sum(axis=0)
    numeric_analysis['N_of_unique_values']   = dataset.apply(pd.Series.nunique)
    numeric_analysis['Unique_values']        = ''
 
    if generate_unique_values: 
        index = 0
        for feature in features:
            numeric_analysis.iloc[index, 6] = np.array_str(dataset[feature].unique())
            index = index + 1

    if generate_csv:
        numeric_analysis.to_csv(path_or_buf='numeric_analysis' + dataset_name  + '.csv')

    return numeric_analysis

def replace_zero_values_to_nan(dataset, feature_names):
    """
    Replace all features present in feature_names with 0 to np.NaN.

    E.g., to be replace afterwards by mean values or dropped.
    """
    import pandas as pd
    import numpy as np
    
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_names, (list, tuple))
    
    for feature_name in feature_names:

        if feature_name in dataset:
            dataset[feature_name].replace(0, np.NaN, inplace=True)
        else:
            raise Exception('Error: ' + feature_name + ' not in dataset.')
            
def replace_nan_values_to_constant(dataset, feature_names, constant=0):
    """
    Replace all features present in feature_names with np.NaN to 0.

    Sometimes even by using a methodology to replace NaN values a few remain,
    we can be interesting to replace them to a constant afterwards.
    """
    import pandas as pd
    import numpy as np
    
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_names, (list, tuple))
    assert isinstance(constant, int)

    for feature_name in feature_names:

        if feature_name in dataset:
            dataset[feature_name].replace(np.NaN, constant, inplace=True)

    return dataset

def show_wordcloud(data, title = None):
    """
    Plots a wordcloud of the data.
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=40, 
        scale=3,
        random_state=1).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    return wordcloud

def util_save_model_pkl(model, model_name):
    """
    Save a model from sklearn as a pickle

    Args:
        model (sklearn model) : model of sklearn (can be regression or classification model)
        model_name (string): name of the model file, e.g., folder/model.pkl
    """
    assert isinstance(model_name, object)

    from sklearn.externals import joblib
    joblib.dump(model, model_name)