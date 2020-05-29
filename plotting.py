# Import all global variables and basic libraries
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURE_SIZE = (6,4)

#############################################################################################################
# PLOTTING METHODS
#############################################################################################################

def get_decision_tree_image(model, features, save_png=True):
    """
    Show the tree of tree classifier and save the tree in png
    """
    from graphviz import Source
    from sklearn import tree
    graph = Source(tree.export_graphviz(model,
                                        out_file=None,
                                        feature_names=features,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True,
                                        class_names=['bad', 'good']))
    png_bytes = graph.pipe(format='png')

    if save_png:
        with open('dtree_pipe.png', 'wb') as tree_file:
            tree_file.write(png_bytes)
    return graph

def plot_cm2inch(*tupl):
    """
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    """

    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    return tuple(i/inch for i in tupl)

def plot_all_two_features_scatter_plot(dataset, feature_names):
    """
    This method generates scatter plots for all combinations of features present in feature_names which is a list.
    Useful if there is a large number of features to inspect the scatter plots, find correlations, outliers etc.
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_names, (list, tuple))

    for feature_name1 in feature_names:
        if feature_name1 not in dataset:
            log_print('Error: ' + feature_name1 + ' not in dataset.')
            continue

        for feature_name2 in feature_names:
            if feature_name1 == feature_name2:
                continue

            if feature_name2 not in dataset:
                log_print('Error: ' + feature_name2 + ' not in dataset.')
                continue

            plot_two_features_scatter(dataset, feature_name1, feature_name2)

def plot_two_features_scatter(dataset, feature_x, feature_y, label, xlim=None, ylim=None):
    """
    Args:
        dataset: Dataset with features to plot
        featureX: Feature in axis X
        featureY: Feature in axis Y
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_x, str)
    assert isinstance(feature_y, str)
    assert xlim is None or isinstance(xlim, (int, float))
    assert ylim is None or isinstance(ylim, (int, float))


    plt.scatter(dataset[dataset[label] == 0][feature_x], dataset[dataset[label] == 0][feature_y],
                marker='.', c='blue', label='negatitve')
    plt.scatter(dataset[dataset[label] == 1][feature_x], dataset[dataset[label] == 1][feature_y],
                marker='.', c='black', label='positive')
    plt.legend()

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

def plot_classifier_comparison(cross_validation_results, names):
    """
    Boxplot the results of cross validation of any classifiers to visual comparison
    """
    fig = plt.figure()
    fig.suptitle('Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(cross_validation_results)
    ax.set_xticklabels(names)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
    
def plot_graphical_exploratory_datanalysis(dataset, figsize=FIGURE_SIZE):
    """
    Plots the correlation matrix, histogram and a heatmap.
    """

    assert isinstance(dataset, pd.core.frame.DataFrame)

    print('Correlation Matrix')
    print(dataset.corr())

    sns.set(style="ticks")
    sns.pairplot(dataset, hue="label")

    dataset.hist()
    dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
    plt.show()

    g = sns.PairGrid(dataset)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
    ax = sns.heatmap(dataset, annot=True, diag_names=True)

def plot_heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    """
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    """

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(plot_cm2inch(figure_width, figure_height))

def plot_boxplot_and_density_individually_with_seaborn(dataset, feature_names, figsize=FIGURE_SIZE):
    """
    Seaborn produces high quality plots.
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)

    # To generate publication quality plots:
    sns.set_style("whitegrid")
    sns.set_context("paper")

    for name in feature_names:
        if name in dataset:
            plt.figure(figsize=(10, 8))
            plt.subplot(211)
            plt.xlim(dataset[name].min(), dataset[name].max()*1.1)
            ax = dataset[name].plot(kind='kde')
            plt.subplot(212)
            plt.xlim(dataset[name].min(), dataset[name].max()*1.1)
            sns.boxplot(x=dataset[name])

def plot_boxplot_and_histogram_individually(dataset, feature_names, figsize=FIGURE_SIZE):

    """
    Plots a boxplot and histograme for feature names contained in feature_names of dataset.
    """

    assert isinstance(dataset, pd.core.frame.DataFrame)

    for name in feature_names:
        if name in dataset:
            dataset.boxplot(column=name, return_type='axes')
            plt.show()
            dataset.hist(column=name, bins=50)
            plt.show()

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu', figsize=FIGURE_SIZE)):
    """
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    """
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2:(len(lines)-2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        log_print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    plot_heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels,
                 figure_width, figure_height, correct_orientation, cmap=cmap)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, figsize=FIGURE_SIZE):
    """
    This function prints and plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/modeselection/plot_confusion_matrix.html#sphx-glr-auto-examples-
    model-selection-plot-confusion-matrix-py
    """
    
    print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_correlation_with_feature(dataset, feature_name, figsize=FIGURE_SIZE):
    """
    This methods plot the correlation between all numeric features with feature_name
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_name, object)
    assert isinstance(figsize, tuple)

    if feature_name in dataset:
        corr = dataset.corr()
        corr = corr.sort_values(feature_name, ascending=False)
        plt.figure(figsize=figsize)
        sns.barplot(corr[feature_name][1:], corr.index[1:], orient='h')

def plot_decision_boundary(X_train, y_train, model, name, figsize=FIGURE_SIZE):
    """
    Plots a decision boundary for a classification problem. The number of componentes is reduced to two
    using TruncatedSVD.
    """

    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.DataFrame)
    assert isinstance(name, object)

    # Import TruncatedSVD: Sparse matrix
    from sklearn.decomposition import TruncatedSVD

    # Create TruncatedSVD instance: model
    svd_model = TruncatedSVD(n_components=2)

    # Apply the fit_transform method of model to grains: pcfeatures
    svd_features = svd_model.fit_transform(X_train)

    # Assign 0th column of pcfeatures: xs
    xs = svd_features[:, 0]
    X = svd_features

    # Assign 1st column of pcfeatures: ys
    ys = svd_features[:, 1]

    # For countour plot
    x_min, x_max = xs.min() - 1, xs.max() + 1
    y_min, y_max = ys.min() - 1, ys.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

    classifier_svd = model
    classifier_svd.fit(X, y_train)

    Z = classifier_svd.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, color='blue')
    plt.title("Decision Boundary {}".format(name))
    plt.scatter(xs, ys, c=y_train, s=40)

def plot_decision_region(X, y, model, features, figsize=FIGURE_SIZE):
    """
    Plot the decision region of the model for two features
    """
    from mlxtend.plotting import plot_decision_regions

    plt.figure(figsize=figsize)
    plt.title('Decision Regions')
    i = 331
    for f1, f2 in zip(*[iter(features)]*2):
        x_feat = X[[f1, f2]]
        model.fit(x_feat, y)
        plt.subplot(i)
        i = i+1

        # Plotting decision regions
        plot_decision_regions(x_feat.as_matrix(), y.as_matrix(), clf=model, res=0.02)
        # Adding axes annotations
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f1+','+f2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_decision_region_grid(x, y, features, classifiers, labels):
    """
    Plot decision regions as a 4x4 grid.
    """
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.gridspec as gridspec

    feature1 = features[0]
    feature2 = features[1]

    gs = gridspec.GridSpec(2, 2)

    _ = plt.figure(figsize=(10, 8))

    x_feat = x[[feature1, feature2]]

    for clf, lab, grd in zip(classifiers, labels, itertools.product([0, 1], repeat=2)):

        clf.fit(x_feat, y)
        _ = plt.subplot(gs[grd[0], grd[1]])
        _ = plot_decision_regions(x_feat.as_matrix(), y.as_matrix(), clf=clf, legend=2)
        plt.title(lab)

    plt.show()

def plot_distplot(dataset, feature_name, figsize=FIGURE_SIZE):
    """
    Plot distplot (flexibly plot a univariate distribution of observations) of feature from dataset.
    This function combines the matplotlib hist function
    (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot()
    functions. kdeplot() fits and plots a univariate or bivariate kernel density estimate.

    This method also prints the skew and kurt of the feature. Skewness is a measure of the symmetry in a
    distribution. A symmetrical dataset will have a skewness equal to 0. So, a normal distribution will
    have a skewness of 0. Skewness essentially measures the relative size of the two tails.
    Kurtosis is a measure of the combined sizes of the two tails. It measures the amount of probability
    in the tails. The value is often compared to the kurtosis of the normal distribution, which is equal to 3.
    If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution
    (more in the tails). If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution
    (less in the tails). http://www.pythonforfinance.net/2016/04/04/python-skew-kurtosis/

    That is, data sets with high kurtosis tend to have heavy tails, or outliers.
    Data sets with low kurtosis tend to have light tails, or lack of outliers.
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_name, object)
    assert isinstance(figsize, tuple)

    if feature_name in dataset:
        plt.figure(figsize=figsize)
        sns.distplot(dataset[feature_name])
        print("Skewness: %f" % dataset[feature_name].skew(),False)
        print("Kurtosis: %f" % dataset[feature_name].kurt(),False)

def plot_feature_importance(model, feature_names, num_of_features_to_show, figsize=FIGURE_SIZE):
    """
    Plot the importance of features for the model

    Args:
        model (sklearn model): sklearn model that has feature_importances_ attribute
        feature_names (list): list of features from the dataset
        num_of_features_to_show (int): number of features to show in plot
    """

    import sklearn.base

    assert isinstance(feature_names, object)
    assert isinstance(figsize, tuple)

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:num_of_features_to_show]

    plt.figure(figsize=figsize)
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", align="center")
    plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()

def plot_generate_all_combinations_of_two_features_scatter_plot(dataset, feature_names, figsize=FIGURE_SIZE):
    """
    This method generates scatter plots for all combinations of features present in feature_names which is a list.
    Useful if there is a large number of features to inspect the scatter plots, find correlations, outliers etc.
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_names, (list, tuple))

    for feature_name1 in feature_names:
        if feature_name1 not in dataset:
            print('Error: ' + feature_name1 + ' not in dataset.')
            continue

        for feature_name2 in feature_names:
            if feature_name1 == feature_name2:
                continue

            if feature_name2 not in dataset:
                print('Error: ' + feature_name2 + ' not in dataset.')
                continue

            plot_two_features_scatter(dataset, feature_name1, feature_name2)

def plot_latitude_and_longitude_on_map(dataset, latitude_feature, longitude_feature, map_location='../images/california.png', figsize=FIGURE_SIZE):
    """
    Plot latitude and longitude data over california map picture

    Args:
        dataset (DataFrame): data with latitude and longitude
        latitude_feature (string): name of the columns of dataset with latitude
        longitude_feature (string): name of the columns of dataset with longitude
    """
    import matplotlib.image as mpimg

    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(latitude_feature, object)
    assert isinstance(longitude_feature, object)
    assert isinstance(map_location, object)
    assert isinstance(figsize, tuple)

    californiimg = mpimg.imread(map_location)
    ax = dataset.plot(kind="scatter", x=longitude_feature, y=latitude_feature, figsize=figsize,
                      cmap=plt.get_cmap("jet"), colorbar=False, alpha=0.4)
    plt.imshow(californiimg, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

# http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
def plot_learning_curve(X_train, y_train, model, modename, figsize=FIGURE_SIZE):
    """
    Plots a learning curve for a especific model and datasets X_train and y_train.
    """


    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.DataFrame)
    assert isinstance(modename, object)

    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure()
    plt.title("Learning Curve {}".format(modename))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_precision_and_recall_curve(fitted_model, X_test, y_test):

    """
    Plots precision and recall curve.

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve
    """

    y_score = fitted_model.predict_proba(X_test)
    y_score = y_score[:, 1]

    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)

    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

def plot_scatter_against_feature(dataset, feature_name, figsize=g_figsize_large, reg_fit=True):
    """
    This method plots scatter between all numeric features against feature_name
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_name, object)
    assert isinstance(reg_fit, bool)

    if feature_name in dataset:
        numericafeatures = list(dataset.select_dtypes(include=["int64", "float64"]).columns)
        numericafeatures.remove(feature_name)

        f = pd.melt(dataset, id_vars=[feature_name], value_vars=sorted(numericafeatures))
        g = sns.FacetGrid(f, col='variable', col_wrap=3, sharex=False, sharey=False)
        plt.xticks(rotation='vertical')
        g = g.map(sns.regplot, 'value', feature_name, scatter_kws={'alpha':0.3}, fit_reg=reg_fit)
        [plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
        plt.figure(figsize=figsize)
        g.fig.tight_layout()
        plt.show()

def plot_scatter_and_boxplot(dataset, label, feature, figsize=FIGURE_SIZE):
    """
    This method plots scatter between the feature and the label and the boxplot of the feature
    """
    
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.scatter(dataset[feature], dataset[label])
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dataset[feature])

def plot_two_features_scatter(dataset, featureX, featureY, arrayX=None, arrayY=None, xlim=None, ylim=None, classification=True):
    """
    Args:
        dataset: Dataset with features to plot
        featureX: Feature in axis X
        featureY: Feature in axis Y
        arrayX, arrayY: These properties can be several points which represent fore example a fitted regression function
                    to be plot on top of the scatter plot.
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(featureX, str)
    assert isinstance(featureY, str)
    assert arrayX is None or isinstance(arrayX, (list, tuple))
    assert arrayY is None or isinstance(arrayY, (list, tuple))
    assert xlim is None or isinstance(xlim, (int, float))
    assert ylim is None or isinstance(ylim, (int, float))
    assert isinstance(classification, bool)

    if arrayX is not None and arrayY is not None:
        assert len(arrayX) == len(arrayY)

    if classification is True:
        plt.scatter(dataset[dataset[g_label] == 0][featureX], dataset[dataset[g_label] == 0][featureY],
                    marker='.', c='blue', label='negatitve')
        plt.scatter(dataset[dataset[g_label] == 1][featureX], dataset[dataset[g_label] == 1][featureY],
                    marker='.', c='black', label='positive')
        plt.legend()
    else:
        plt.scatter(dataset[featureX], dataset[featureY], marker='.')

    if arrayX is not None and arrayY is not None:
        plt.plot(arrayX, arrayY, color='black', linewidth=3)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.show()

def plot_two_value_counts_barh(dataset, feature_names, quantity_data=-1, figsize=FIGURE_SIZE):
    """
    This method plots the value counts of two or more features
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_names, object)
    assert isinstance(quantity_data, int)
    assert isinstance(figsize, tuple)

    num_feat = len(feature_names)
    fig, axs = plt.subplots(ncols=num_feat, figsize=figsize)
    fig.subplots_adjust(wspace=0.5)

    for i, feature in enumerate(feature_names):
        if feature in dataset:
            if quantity_data == -1:
                dataset[feature].value_counts().plot(kind='barh', figsize=figsize, ax=axs[i])
            else:
                dataset[feature].value_counts()[:quantity_data].plot(kind='barh', figsize=figsize, ax=axs[i])

def plot_validation_curve(x_train, y_train, model, model_name, params):
    """
    Plot the validation curve by determining training and test scores for varying parameters values
    """
    assert isinstance(x_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, (pd.core.frame.DataFrame, pd.core.series.Series))

    from sklearn.model_selection import validation_curve

    param_name = params[0]
    param_range = params[1]
    train_scores, test_scores = validation_curve(model, x_train, y_train, param_name,
                                                 param_range, cv=5, scoring='f1')

    train_scores_mean = np.mean(train_scores, axis=1)
    #train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve {}".format(model_name))
    plt.xlabel(param_name)
    plt.ylabel("F1-Score")
    plt.ylim(0.0, 1.1)
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy")
    plt.legend(loc="best")
    plt.xticks(param_range)
    plt.show()

def plot_value_counts_barh(dataset, feature_name, quantity_data=-1, figsize=FIGURE_SIZE):
    """
    This method plots the value counts of one feature
    """
    assert isinstance(dataset, pd.core.frame.DataFrame)
    assert isinstance(feature_name, object)
    assert isinstance(quantity_data, int)
    assert isinstance(figsize, tuple)
    
    if feature_name in dataset:
        if quantity_data == -1:
            dataset[feature_name].value_counts().plot.barh(figsize=figsize)
        else:
            dataset[feature_name].value_counts()[:quantity_data].plot.barh(figsize=figsize)

def show_values(pc, fmt="%.2f", **kw):
    """
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    """
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
