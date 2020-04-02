import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from definitions import RESULT_DIR


def plot_marginals(real, synthetic, dataset, model):
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

    basepath = os.path.join(RESULT_DIR, dataset, model)
    filepath = os.path.join(basepath, '{0}_{1}_c_marginals.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
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

    basepath = os.path.join(RESULT_DIR, dataset, model)
    filepath = os.path.join(basepath, '{0}_{1}_d_marginals.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    plt.savefig(filepath)
