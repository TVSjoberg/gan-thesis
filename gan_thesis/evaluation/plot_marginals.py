import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import kde
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


