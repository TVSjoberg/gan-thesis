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


def plot_all_marginals(dataset, data, force=True, pass_tgan=True):
    real = dataset.train
    cols = real.columns

    alist = data.split(sep='-', maxsplit=1)
    base_path = os.path.join(RESULT_DIR, *alist)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    samples_wgan = dataset.samples.get('wgan')
    samples_ctgan = dataset.samples.get('ctgan')
    if pass_tgan:
        samples_tgan = dataset.samples.get('tgan')
        samples = [samples_wgan, samples_ctgan, samples_tgan]
        models = ['WGAN', 'CTGAN', 'TGAN']
    else:
        samples = [samples_wgan, samples_ctgan]
        models = ['WGAN', 'CTGAN']

    i_cont = real.columns.get_indexer(real.select_dtypes(np.number).columns)
    i_cat = [i for i in range(len(cols)) if i not in i_cont]

    # Plot a picture of all continuous columns in a (,3) grid with all models combined
    j = 0
    cols = 3
    rows = np.ceil(len(i_cont) / cols)
    plt.figure(figsize=(15, 10))
    for i in i_cont:
        j += 1
        plt.subplot(rows, cols, j)
        sns.distplot(real.iloc[:, i], label="Real")
        for k in range(len(samples)):
            sns.distplot(samples[k].iloc[:, i], label=models[k])
        plt.legend()

    file_path = os.path.join(base_path, '{0}_combined_c_marginals.png'.format(data))
    plt.savefig(file_path)

    # Plot a picture of all continuous columns in a (,3) grid with all models combined
    j = 0
    cols = 3
    rows = np.ceil((len(i_cont) / cols)*len(models))
    plt.figure(figsize=(15, 10))
    for k in range(len(models)):
        for i in i_cont:
            j += 1
            plt.subplot(rows, cols, j)
            sns.distplot(real.iloc[:, i], label="Real")
            sns.distplot(samples[k].iloc[:, i], label=models[k])
            plt.legend()

    file_path = os.path.join(base_path, '{0}_separated_c_marginals.png'.format(data))
    plt.savefig(file_path)

    temp = real.copy()
    wgan = samples_wgan.copy()
    ctgan = samples_ctgan.copy()

    identifier = ['Real'] * len(temp)
    temp['Synthetic'] = identifier
    identifier = ['WGAN'] * len(wgan)
    wgan['Synthetic'] = identifier
    identifier = ['CTGAN'] * len(ctgan)
    ctgan['Synthetic'] = identifier

    if pass_tgan:
        tgan = samples_tgan.copy()
        identifier = ['TGAN'] * len(tgan)
        tgan['Synthetic'] = identifier
        frames = [temp, wgan, tgan, ctgan]
        result = pd.concat(frames)
    else:
        frames = [temp, wgan, ctgan]
        result = pd.concat(frames)

    j = 0
    cols = 3
    rows = int(np.ceil(len(i_cat) / cols))
    f, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    for i in i_cat:
        temp = result[result['Synthetic'] == 'Real'].iloc[:, i]
        vals = temp.value_counts(normalize=True)
        id = ['Real'] * len(vals)
        vals_id = list(zip(vals.index, vals, id))
        rel_counts = pd.DataFrame(vals_id, columns=['Feature', 'Frequency', 'Model'])

        for model in models:
            temp = result[result['Synthetic'] == model].iloc[:, i]
            vals = temp.value_counts(normalize=True)
            id = [model]*len(vals)
            vals_id = list(zip(vals.index, vals, id))
            rel_counts = rel_counts.append(pd.DataFrame(vals_id, columns=['Feature', 'Frequency', 'Model']), ignore_index=True)

        sns.barplot(x='Feature', y='Frequency', hue='Model', data=rel_counts, ax=axes[j])

        # (result
        #  .groupby(x)[y]
        #  .value_counts(normalize=True)
        #  .mul(100)
        #  .rename('percent')
        #  .reset_index()
        #  .pipe((sns.catplot, 'data'), x=x, y='percent', hue=y, kind='bar', ax=axes[j]))
        # sns.countplot(x=real.columns.tolist()[i], data=result, hue='Synthetic', ax=axes[j])
        if (j-1) % 3 == 0:
            axes[j].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                      ncol=4, fancybox=True, shadow=True)
        else:
            axes[j].get_legend().remove()
        j += 1

    filepath = os.path.join(base_path, '{0}_all_d_marginals.png'.format(data))
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)
    plt.savefig(filepath)
    print(['saved figure at',filepath])


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
