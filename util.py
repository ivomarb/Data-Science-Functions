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