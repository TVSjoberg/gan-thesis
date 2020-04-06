from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np


def association(dataset, split=False):
    data = dataset.data
    discrete_columns, continuous_columns = dataset.get_columns()
    columns = data.columns.to_list()

    association_matrix = association_real = np.ones(shape=(len(columns),len(columns)))

    for i in range(len(columns)):
        for j in range(i):
            if (columns[i] in continuous_columns) and (columns[j] in continuous_columns):
                association_matrix[i, j] = spearmanr(data.iloc[:, i], data.iloc[:, j])[0]
            if (columns[i] in discrete_columns) and (columns[j] in discrete_columns):
                association_matrix[i, j] = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
            if (columns[i] in continuous_columns) and (columns[j] in discrete_columns):
                association_matrix[i, j] = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j],
                                                                    bin_axis=[True, False])
            if (columns[i] in discrete_columns) and (columns[j] in continuous_columns):
                association_matrix[i, j] = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j],
                                                                    bin_axis=[False, True])


def mutual_info_score_binned(x, y, bin_axis=None, bins=10):
    if bin_axis is None:
        bin_axis = [True, False]  # Bin x, don't bin y

    x = pd.cut(x, bins=bins) if bin_axis[0] else x
    y = pd.cut(y, bins=bins) if bin_axis[1] else y
    return mutual_info_score(x, y)
