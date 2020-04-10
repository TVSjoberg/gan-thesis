import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return f1_score(test_y, test, average='micro')
    elif metric == "accuracy":
        return accuracy_score(test_y, test)
    elif metric == "r2":
        return [r2_score(test_y, test)]
    else:
        raise Exception("Metric not recognized.")


def predictions_by_dimension(train, test, discrete_columns, continuous_columns):
    features = train.columns.to_list()
    methods = ["random_forest", "adaboost", "regression"] # mlp also available
    prediction_scores = pd.DataFrame(index=methods, columns=features)

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
            prediction_scores.loc[:, features[feature_index]] = temp_scores
        elif features[feature_index] in continuous_columns:
            temp_scores = []
            for method in methods:
                temp_scores.append(prediction_score(
                    one_hot_train, train.iloc[:, feature_index],
                    one_hot_test, test.iloc[:, feature_index],
                    metric="r2", model=method, target='continuous'
                ))
                print(temp_scores)
            prediction_scores.loc[:, features[feature_index]] = temp_scores
    return prediction_scores


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = intercept + slope * x
    plt.plot(x, y, '--')


def plot_predictions_by_dimension(real, samples, data_test, discrete_columns, continuous_columns,
                                  dataset, model, force=True):
    score_y_by_dimension = predictions_by_dimension(samples, data_test, discrete_columns, continuous_columns)
    score_x_by_dimension = predictions_by_dimension(real, data_test, discrete_columns, continuous_columns)
    mean_x_by_dimension = score_x_by_dimension.mean(axis=0)
    mean_y_by_dimension = score_y_by_dimension.mean(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(mean_x_by_dimension, mean_y_by_dimension)
    ax.set_title("Machine Learning Efficiency")
    ax.set_ylabel("Sample features")
    ax.set_xlabel("Real features")
    abline(1, 0)

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
