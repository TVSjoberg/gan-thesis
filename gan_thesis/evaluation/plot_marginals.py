import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import kde
import os
from definitions import RESULT_DIR
from gan_thesis.data.load_data import *


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

    
    
def contour_grid(df, title, nbins = 50, contour = True):
    c = df.columns
    num_var = len(c)
    plt.figure(figsize = (15,10))
    for i in range(num_var):
        for j in range(num_var):
            if  j<i:
                plt.subplot(num_var, num_var, 1+num_var*i + j)
                countour_2d_plt(df[c[i]],df[c[j]], nbins, contour)
    plt.show()

def kde_calc(x, y, nbins = 20):
    k = kde.gaussian_kde((x,y))
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return xi, yi, zi

def countour_2d_plt(x, y, nbins = 50, contour = True):
    xi, yi, zi = kde_calc(x, y, nbins)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='coolwarm')
    if contour:
        plt.contour(xi,yi, zi.reshape(xi.shape), colors = 'black')
    return (xi, yi, zi)



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

    temp = real.copy()
    wgan = samples_wgan.copy()
    tgan = samples_tgan.copy()
    ctgan = samples_ctgan.copy()
    listofzeros = ['Real'] * len(real)
    listofones = ['WGAN'] * len(wgan)
    listoftwos = ['CTGAN'] * len(ctgan)
    listofthrees = ['TGAN'] * len(tgan)
    temp['Synthetic'] = listofzeros
    wgan['Synthetic'] = listofones
    ctgan['Synthetic'] = listoftwos
    tgan['Synthetic'] = listofthrees
    frames = [temp, wgan, tgan, ctgan]
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

    filepath = os.path.join(basepath, '{0}_all_d_marginals.png'.format(data))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)


def plot_cond_marginals():

    dataset = load_data('cond_cat-test1')
    wgan_samples = dataset.samples.get('wgan')
    ctgan_samples = dataset.samples.get('ctgan')
    tgan_samples = dataset.samples.get('tgan')
    cc_real = dataset.data

    labels = ['ind_feat_0_label_0', 'ind_feat_0_label_1', 'ind_feat_0_label_2']
    cond_labels = ['cond_feat_00_label_0', 'cond_feat_00_label_1', 'cond_feat_00_label_2']

    wgan_cond = [wgan_samples['ind_feat_0'] == labels[i] for i in range(3)]
    wgan = [wgan_samples['cond_feat_00'][wgan_cond[i]] for i in range(3)]
    ctgan_cond = [ctgan_samples['ind_feat_0'] == labels[i] for i in range(3)]
    ctgan = [ctgan_samples['cond_feat_00'][ctgan_cond[i]] for i in range(3)]
    tgan_cond = [tgan_samples['ind_feat_0'] == labels[i] for i in range(3)]
    tgan = [tgan_samples['cond_feat_00'][tgan_cond[i]] for i in range(3)]

    real_cond = [cc_real['ind_feat_0'] == labels[i] for i in range(3)]
    real = [cc_real['cond_feat_00'][real_cond[i]] for i in range(3)]

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    sns.countplot(wgan[0], order=cond_labels)
    plt.subplot(132)
    sns.countplot(wgan[1], order=cond_labels)
    plt.subplot(133)
    sns.countplot(wgan[2], order=cond_labels)
    plt.suptitle('ctgan')
    plt.show()
