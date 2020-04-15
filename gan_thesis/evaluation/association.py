from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from definitions import RESULT_DIR
from gan_thesis.data.load_data import Dataset


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


def association_difference(real=None, samples=None, association_real=None, association_samples=None):
    if (association_real is None) or (association_samples is None):
        association_real = association(real)
        association_samples = association(samples)

    return euclidean(association_real.to_numpy().flatten(), association_samples.to_numpy().flatten())


def plot_association(real_dataset, samples, dataset, model, force=True):
    association_real = association(real_dataset)
    samples_dataset = Dataset(None, None, samples, real_dataset.info, None)
    association_samples = association(samples_dataset)

    mask = np.triu(np.ones_like(association_real, dtype=np.bool))

    colormap = sns.diverging_palette(20, 220, n=256)

    plt.figure(figsize=(20, 10))
    plt.suptitle(model.upper() + ' Association')
    plt.subplot(1, 2, 1)
    plt.title('Real')
    sns.heatmap(association_real,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap)

    plt.subplot(1, 2, 2)
    plt.title('Samples')
    sns.heatmap(association_samples,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap)

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist, model)
    filepath = os.path.join(basepath, '{0}_{1}_association.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    return association_difference(association_real=association_real, association_samples=association_samples)


def plot_all_association(complete_dataset, dataset, force=True):
    association_real = association(complete_dataset)

    samples_wgan = complete_dataset.samples.get('wgan')
    samples_tgan = complete_dataset.samples.get('tgan')
    samples_ctgan = complete_dataset.samples.get('ctgan')

    samples_dataset = Dataset(None, None, samples_wgan, complete_dataset.info, None)
    association_wgan = association(samples_dataset)
    samples_dataset = Dataset(None, None, samples_ctgan, complete_dataset.info, None)
    association_ctgan = association(samples_dataset)
    samples_dataset = Dataset(None, None, samples_tgan, complete_dataset.info, None)
    association_tgan = association(samples_dataset)

    mask = np.triu(np.ones_like(association_real, dtype=np.bool))

    colormap = sns.diverging_palette(20, 220, n=256)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 6))
    cbar_ax = fig.add_axes([.95, .3, .02, .4])
    plt.tight_layout()
    ax1.set_title('Real')
    ax1.set_aspect('equal')

    chart = sns.heatmap(association_real,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap,
                ax=ax1,
                cbar=False)

    chart.set_yticklabels(labels=chart.get_yticklabels(), rotation=0)

    ax2.set_title('WGAN')
    ax2.set_aspect('equal')

    sns.heatmap(association_wgan,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap,
                ax=ax2,
                cbar=False)

    ax3.set_title('CTGAN')
    ax3.set_aspect('equal')

    sns.heatmap(association_ctgan,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap,
                ax=ax3,
                cbar=False)

    ax4.set_title('TGAN')
    ax4.set_aspect('equal')

    sns.heatmap(association_tgan,
                vmin=-1,
                vmax=None,
                mask=mask,
                annot=False,
                cmap=colormap,
                ax=ax4,
                cbar=True,
                cbar_ax=cbar_ax)

    plt.subplots_adjust(wspace=0.1)

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist)
    filepath = os.path.join(basepath, '{0}_all_association.png'.format(dataset))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()
