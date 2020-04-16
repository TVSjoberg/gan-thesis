import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from definitions import RESULT_DIR


def plot_marginals(real, synthetic, dataset, model, force=True):
    cols = synthetic.columns

    i_cont = real.columns.get_indexer(real.select_dtypes(np.number).columns)
    i_cat = [i for i in range(len(cols)) if i not in i_cont]

    j = 0
    cols = 3
    rows = np.ceil(len(i_cont) / cols)
    plt.figure(figsize=(15, 10))
    for i in i_cont:
        j += 1
        plt.subplot(rows, cols, j)
        sns.distplot(synthetic.iloc[:, i], label='Synthetic')
        sns.distplot(real.iloc[:, i], label="Real")
        plt.legend()

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist, model)
    filepath = os.path.join(basepath, '{0}_{1}_c_marginals.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)

    temp = real.copy()
    temp2 = synthetic.copy()
    listofzeros = ['Real'] * len(real)
    listofones = ['Synthetic'] * len(synthetic)
    temp['Synthetic'] = listofzeros
    temp2['Synthetic'] = listofones
    frames = [temp, temp2]
    result = pd.concat(frames)

    j = 0
    cols = 3
    rows = np.ceil(len(i_cat) / cols)
    plt.figure(figsize=(15, 10))
    for i in i_cat:
        j += 1
        plt.subplot(rows, cols, j)
        sns.countplot(x=real.columns.tolist()[i], data=result, hue='Synthetic')
        plt.legend()

    filepath = os.path.join(basepath, '{0}_{1}_d_marginals.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)


def plot_all_marginals(dataset, data, force=True):
    real = dataset.train
    data_test = dataset.test
    discrete_columns, continuous_columns = dataset.get_columns()
    cols = real.columns

    samples_wgan = dataset.samples.get('wgan')
    samples_tgan = dataset.samples.get('tgan')
    samples_ctgan = dataset.samples.get('ctgan')
    samples = [samples_wgan, samples_ctgan, samples_tgan]
    models = ['wgan', 'ctgan', 'tgan']

    i_cont = real.columns.get_indexer(real.select_dtypes(np.number).columns)
    i_cat = [i for i in range(len(cols)) if i not in i_cont]

    j = 0
    cols = 3
    rows = np.ceil(len(i_cont) / cols)
    plt.figure(figsize=(15, 10))
    for i in i_cont:
        j += 1
        plt.subplot(rows, cols, j)
        sns.distplot(real.iloc[:, i], label="Real")
        sns.distplot(samples[0].iloc[:, i], label=models[0])
        sns.distplot(samples[1].iloc[:, i], label=models[1])
        sns.distplot(samples[2].iloc[:, i], label=models[2])
        plt.legend()

    alist = data.split(sep='-', maxsplit=1)
    basepath = os.path.join(RESULT_DIR)
    filepath = os.path.join(basepath, '{0}_all_c_marginals.png'.format(data))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)

    # temp = real.copy()
    # temp2 = synthetic.copy()
    # listofzeros = ['Real'] * len(real)
    # listofones = ['Synthetic'] * len(synthetic)
    # temp['Synthetic'] = listofzeros
    # temp2['Synthetic'] = listofones
    # frames = [temp, temp2]
    # result = pd.concat(frames)
    #
    # j = 0
    # cols = 3
    # rows = np.ceil(len(i_cat) / cols)
    # plt.figure(figsize=(15, 10))
    # for i in i_cat:
    #     j += 1
    #     plt.subplot(rows, cols, j)
    #     sns.countplot(x=real.columns.tolist()[i], data=result, hue='Synthetic')
    #     plt.legend()
    #
    # filepath = os.path.join(basepath, '{0}_{1}_d_marginals.png'.format(dataset, model))
    # if not os.path.exists(basepath):
    #     os.makedirs(basepath)
    # if os.path.isfile(filepath) and force:
    #     os.remove(filepath)
    # plt.savefig(filepath)
