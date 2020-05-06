import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from definitions import RESULT_DIR


def prediction_score(train_x, train_y, test_x, test_y, metric, model, target):

    models = {
        "continuous": {
            "random_forest": RandomForestRegressor(n_estimators=10),
            "adaboost": AdaBoostRegressor(n_estimators=10),
            "regression": LinearRegression(),
            "mlp": MLPRegressor()
        },
        "discrete": {
            "random_forest": RandomForestClassifier(n_estimators=10),
            "adaboost": AdaBoostClassifier(n_estimators=10),
            "regression": LogisticRegression(),
            "mlp": MLPClassifier()
        },
    }
    m = models[target][model]

    m.fit(train_x, train_y)
    test = m.predict(test_x)
    if metric == "f1":
        return np.max([f1_score(test_y, test, average='micro'), 0])
    elif metric == "accuracy":
        return np.max([accuracy_score(test_y, test), 0])
    elif metric == "r2":
        return np.max([r2_score(test_y, test), 0])
    else:
        raise Exception("Metric not recognized.")


def predictions_by_dimension(train, test, discrete_columns, continuous_columns):
    features = train.columns.to_list()
    methods = ["random_forest", "adaboost", "regression"] # mlp also available
    prediction_scores = pd.DataFrame(index=features, columns=methods)

    for feature_index in range(len(features)):
        # drop target feature
        temp_train = train.drop(train.columns[feature_index], axis=1)
        temp_test = test.drop(test.columns[feature_index], axis=1)
        # one-hot-encode non-target features
        discrete_used_feature_indices = [feature for feature in features if
                                         (feature in discrete_columns) &
                                         (feature != features[feature_index])]
        one_hot_train = pd.get_dummies(temp_train, columns=discrete_used_feature_indices)
        one_hot_test = pd.get_dummies(temp_test, columns=discrete_used_feature_indices)

        # make sure train and test have equal one-hot-encoding space
        if len(one_hot_train.columns) != len(one_hot_test.columns):
            for i in one_hot_train.columns:
                if i not in one_hot_test.columns: one_hot_test[i] = 0
            for i in one_hot_test.columns:
                if i not in one_hot_train.columns: one_hot_train[i] = 0
            # use the same column order for the test set as for train
            one_hot_test = one_hot_test.reindex(one_hot_train.columns, axis=1)

        if features[feature_index] in discrete_columns:
            temp_scores = []
            for method in methods:
                temp_scores.append(prediction_score(
                    one_hot_train, train.iloc[:, feature_index],
                    one_hot_test, test.iloc[:, feature_index],
                    metric="f1", model=method, target='discrete'
                ))
                print(temp_scores)
            prediction_scores.loc[features[feature_index], :] = temp_scores
        elif features[feature_index] in continuous_columns:
            temp_scores = []
            for method in methods:
                temp_scores.append(prediction_score(
                    one_hot_train, train.iloc[:, feature_index],
                    one_hot_test, test.iloc[:, feature_index],
                    metric="r2", model=method, target='continuous'
                ))
                print(temp_scores)
            prediction_scores.loc[features[feature_index], :] = temp_scores
    return prediction_scores


def abline(slope, intercept, ax):
    """Plot a line from slope and intercept"""
    x = np.array(ax.get_xlim())
    y = intercept + slope * x
    ax.plot(x, y, '--')


def plot_predictions_by_dimension(real, samples, data_test, discrete_columns, continuous_columns,
                                  dataset, model, force=True):
    score_y_by_dimension = predictions_by_dimension(samples, data_test, discrete_columns, continuous_columns)
    score_x_by_dimension = predictions_by_dimension(real, data_test, discrete_columns, continuous_columns)
    mean_x_by_dimension = score_x_by_dimension.mean(axis=1)
    mean_y_by_dimension = score_y_by_dimension.mean(axis=1)
    col_type = ['Categorical' if (col in discrete_columns) else 'Continuous' for col in mean_x_by_dimension.index]
    results = pd.DataFrame({'x': mean_x_by_dimension, 'y': mean_y_by_dimension, 'col_type': col_type})

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(x='x', y='y', ax=ax, hue='col_type', data=results)
    ax.set_title("Machine Learning Efficiency")
    ax.set_ylabel("Sample features")
    ax.set_xlabel("Real features")
    abline(1, 0, ax)

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist, model)
    filepath = os.path.join(basepath, '{0}_{1}_ml_efficiency.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)

    score_x_by_dimension.to_csv(os.path.join(basepath, '{0}_{1}_ml_real.csv'.format(dataset, model)), index=True)
    score_y_by_dimension.to_csv(os.path.join(basepath, '{0}_{1}_ml_samples.csv'.format(dataset, model)), index=True)

    return score_x_by_dimension, score_y_by_dimension


def plot_all_predictions_by_dimension(dataset, data):

    real = dataset.train
    data_test = dataset.test
    discrete_columns, continuous_columns = dataset.get_columns()

    samples_wgan = dataset.samples.get('wgan')
    # samples_tgan = dataset.samples.get('tgan')
    samples_ctgan = dataset.samples.get('ctgan')
    # samples = [samples_wgan, samples_ctgan, samples_tgan]
    samples = [samples_wgan, samples_ctgan]
    # models = ['wgan', 'ctgan', 'tgan']
    models = ['wgan', 'ctgan']

    # fig, axn = plt.subplots(1, 3, figsize=(20, 6))
    fig, axn = plt.subplots(1, 2, figsize=(20, 6))
    alist = data.split(sep='-', maxsplit=1)
    basepath = os.path.join(RESULT_DIR, *alist)
    for model in models:
        if not os.path.exists(os.path.join(basepath, model)):
            os.makedirs(os.path.join(basepath, model))

    for i in range(len(models)):
        score_y_by_dimension = predictions_by_dimension(samples[i], data_test, discrete_columns, continuous_columns)
        score_x_by_dimension = predictions_by_dimension(real, data_test, discrete_columns, continuous_columns)
        mean_x_by_dimension = score_x_by_dimension.mean(axis=1)
        mean_y_by_dimension = score_y_by_dimension.mean(axis=1)
        score_x_by_dimension.to_csv(os.path.join(basepath, models[i], '{0}_{1}_ml_real.csv'.format(data, models[i])), index=True)
        score_y_by_dimension.to_csv(os.path.join(basepath, models[i], '{0}_{1}_ml_samples.csv'.format(data, models[i])), index=True)
        col_type = ['Categorical' if (col in discrete_columns) else 'Continuous' for col in mean_x_by_dimension.index]
        results = pd.DataFrame({'x': mean_x_by_dimension, 'y': mean_y_by_dimension, 'col_type': col_type})
        ax = axn[i]
        sns.scatterplot(x='x', y='y', ax=ax, hue='col_type', data=results)
        ax.set_title(models[i])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Sample features")
        ax.set_xlabel("Real features")
        abline(1, 0, ax)

    alist = data.split(sep='-', maxsplit=1)
    # dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist)
    filepath = os.path.join(basepath, '{0}_all_ml_efficiency.png'.format(data))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    plt.savefig(filepath)
