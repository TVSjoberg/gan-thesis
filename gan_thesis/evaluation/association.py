from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from definitions import RESULT_DIR


def association(dataset, split=False):
    data = dataset.data
    discrete_columns, continuous_columns = dataset.get_columns()
    columns = data.columns.to_list()

    association_matrix = np.ones(shape=(len(columns), len(columns)))

    for i in range(len(columns)):
        for j in range(i):
            if (columns[i] in continuous_columns) and (columns[j] in continuous_columns):
                association_matrix[i, j] = pearsonr(data.iloc[:, i], data.iloc[:, j])[0]
            if (columns[i] in discrete_columns) and (columns[j] in discrete_columns):
                association_matrix[i, j] = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
            if (columns[i] in continuous_columns) and (columns[j] in discrete_columns):
                association_matrix[i, j] = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j],
                                                                    bin_axis=[True, False])
            if (columns[i] in discrete_columns) and (columns[j] in continuous_columns):
                association_matrix[i, j] = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j],
                                                                    bin_axis=[False, True])

    return pd.DataFrame(association_matrix, index=columns, columns=columns)


def mutual_info_score_binned(x, y, bin_axis=None, bins=10):
    if bin_axis is None:
        bin_axis = [True, False]  # Bin x, don't bin y

    x = pd.cut(x, bins=bins) if bin_axis[0] else x
    y = pd.cut(y, bins=bins) if bin_axis[1] else y
    return mutual_info_score(x, y)


def association_difference(real=None, samples=None, association_real=None, association_samples=None, force=True):
    if (association_real is None) or (association_samples is None):
        association_real = association(real)
        association_samples = association(samples)

    return np.sum(np.abs((association_real - association_samples).values))


def plot_association(real, samples, dataset, model):
    association_real = association(real)
    association_samples = association(samples)

    mask = np.triu(np.ones_like(association_real, dtype=np.bool))

    plt.figure(figsize=(12, 5))
    plt.suptitle(model.upper() + ' Association')
    plt.subplot(1, 2, 1)
    plt.title('Real')
    sns.heatmap(association_real,
                vmin=0,
                vmax=1,
                mask=mask,
                annot=True,
                cmap='coolwarm')
    plt.subplot(1, 2, 2)
    plt.title('Samples')
    sns.heatmap(association_samples,
                vmin=0,
                vmax=1,
                mask=mask,
                annot=True,
                cmap='coolwarm')

    basepath = os.path.join(RESULT_DIR, dataset, model)
    filepath = os.path.join(basepath, '{0}_{1}_association.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)

    plt.savefig(filepath)

    return association_difference(association_real=association_real, association_samples=association_samples)
