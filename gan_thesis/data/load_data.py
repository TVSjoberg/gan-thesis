import pandas as pd
import os
import json
import shutil
import numpy as np
from gan_thesis.data.datagen import *
from definitions import DATA_DIR, ROOT_DIR, mvn_test1, mvn_test2, mvn_mix_test1


class Dataset:
    def __init__(self, train, test, data, info, samples):
        self.train = train
        self.test = test
        self.data = data
        self.info = info
        self.samples = samples

    def get_columns(self):
        d_col = self.info.get('discrete_columns')
        c_col = self.info.get('continuous_columns')
        return d_col, c_col


def load_data(dataset, data_params=None):
    '''
    :param dataset: A string with the following structure: 'dataset-identifier'. Dataset tells us which dataset to load,
                    identifier allows us to specify different tests. Example: mvn-test1
    :param data_params: Required for synthetic datasets
    :return: A Dataset variable
    '''
    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    pathname = os.path.join(DATA_DIR, *alist)
    filelist = ['train.csv', 'test.csv', 'data.csv', 'info.json']
    filelist = map(lambda x: os.path.join(pathname, x), filelist)
    if not all([os.path.isfile(f) for f in filelist]):
        if os.path.exists(pathname):
            shutil.rmtree(pathname)
        os.makedirs(pathname)
        load_wrapper[dataset](pathname, data_params)

    train = pd.read_csv(os.path.join(pathname, 'train.csv'))
    test = pd.read_csv(os.path.join(pathname, 'test.csv'))
    df = pd.read_csv(os.path.join(pathname, 'data.csv'))
    samples_dir = {}
    for model in ['ctgan', 'tgan', 'wgan']:
        fname = os.path.join(pathname, model, '{0}_{1}_samples.csv'.format(dataset, model))
        if os.path.isfile(fname):
            samples = pd.read_csv(fname)
            samples_dir[model] = samples

    with open(os.path.join(pathname, 'info.json'), "r") as read_file:
        info = json.load(read_file)

    return Dataset(train, test, df, info, samples_dir)


def load_adult(dirname, *args):
    n_test = 9600  # Same as CTGAN paper
    info = {
        "columns": ['age',
                    'workclass',
                    'fnlwgt',
                    'education',
                    'education-num',
                    'marital-status',
                    'occupation',
                    'relationship',
                    'race',
                    'sex',
                    'capital-gain',
                    'capital-loss',
                    'hours-per-week',
                    'native-country',
                    'income'],

        "discrete_columns": ['workclass',
                             'education',
                             'marital-status',
                             'occupation',
                             'relationship',
                             'race',
                             'sex',
                             'native-country',
                             'income'],

        "continuous_columns": ['age',
                               'fnlwgt',
                               'education-num',
                               'capital-gain',
                               'capital-loss',
                               'hours-per-week'],
        "n_test": n_test,
        "identifier": 'adult'
    }

    cc = info.get('columns')
    df = pd.read_csv(os.path.join(ROOT_DIR, 'adult.csv'), names=cc, header=0)
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    #                 names=cc, header=0)
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df=df, n_test=n_test)
    df.to_csv(os.path.join(dirname, 'data.csv'), index=False)
    train.to_csv(os.path.join(dirname, 'train.csv'), index=False)
    test.to_csv(os.path.join(dirname, 'test.csv'), index=False)

    with open(os.path.join(dirname, 'info.json'), "w") as write_file:
        json.dump(info, write_file)


def load_mvn_mixture(pathname, data_params):
    n_samples = data_params['n_samples']
    proportions = data_params['proportions']
    means = data_params['means']
    corrs = data_params['corrs']
    var = data_params['vars']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = mixture_gauss(n_samples, proportions, means, var, corrs, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []

    save_data(df, info, pathname)


def load_mvn(pathname, data_params):
    n_samples = data_params['n_samples']
    mean = data_params['mean']
    corr = data_params['corr']
    var = data_params['var']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = multivariate_df(n_samples, mean, var, corr, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []
    save_data(df, info, pathname)



def load_ln_mixture(pathname, data_params):
    n_samples = data_params['n_samples']
    proportions = data_params['proportions']
    means = data_params['means']
    corrs = data_params['corrs']
    var = data_params['vars']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = mixture_log_normal(n_samples, proportions, means, var, corrs, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []

    save_data(df, info, pathname)


def load_multinomial(pathname, data_params):
    n_samples = data_params.get('n_samples')
    probabilities = data_params.get('probabilities')
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')
    
    df, info = multinomial(n_samples, probabilities, seed=seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    save_data(df, info, pathname)


def load_cond_multinomial(pathname, data_params):
    n_samples = data_params.get('n_samples')
    ind_probs = data_params.get('ind_probs')
    cond_probs = data_params.get('cond_probs')
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')
    
    df, info= multinomial_cond(n_samples, ind_probs, cond_probs, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    save_data(df, info, pathname)


def load_gauss_cond(pathname, data_params):
    n_samples = data_params.get('n_samples')
    ind_probs = data_params.get('ind_probs')
    cond_probs = data_params.get('cond_probs')

    if cond_probs is None:
        mode = 'ind_cat'
    else:
        mode = 'cond_cat'
    means = data_params.get('means')
    corrs = data_params.get('corrs')
    var = data_params.get('vars')

    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    if mode == 'ind_cat':
        cond_df, cond_info =  multinomial(n_samples, ind_probs, seed = seed)
    else:
        cond_df, cond_info = multinomial_cond(n_samples, ind_probs, cond_probs, seed)

    df, info = cat_mixture_gauss(cond_df, cond_info, means, var, corrs, seed)

    info['seed'] = seed
    info['continuous_columns'] = [f for f in  df.columns.to_list() if f not in cond_df.columns.to_list()]
    info['discrete_columns'] = cond_df.columns.to_list()
    save_data(df, info, pathname)



def load_ln(pathname, data_params):
    n_samples = data_params['n_samples']
    mean = data_params['mean']
    cov = data_params['cov']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = log_normal_df(n_samples, mean, cov, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []

    save_data(df, info, pathname)


def save_data(df, info, dirname):

    df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df=df, n_test=int(np.floor(0.1 * len(df))))
    df.to_csv(os.path.join(dirname, 'data.csv'), index=False)
    train.to_csv(os.path.join(dirname, 'train.csv'), index=False)
    test.to_csv(os.path.join(dirname, 'test.csv'), index=False)

    with open(os.path.join(dirname, 'info.json'), "w") as write_file:
        json.dump(info, write_file)


def train_test_split(df, n_test):
    assert n_test < len(df), "n_test larger than n_tot"
    test = df.sample(n=n_test)
    train = df.drop(test.index)

    return train, test


def save_samples(df, dataset, model, force=True):
    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    fname = os.path.join(DATA_DIR, *alist, model, '{0}_{1}_samples.csv'.format(dataset, model))
    if os.path.isfile(fname) and not force:
        return
    base_path = os.path.dirname(fname)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    df.to_csv(fname, index=False)


load_wrapper = {
    'adult': load_adult,
    'mvn': load_mvn,
    'mvn-mixture': load_mvn_mixture,
    'ln': load_ln,
    'ln-mixture': load_ln_mixture,
    'cat': load_multinomial,
    'cond-cat' : load_cond_multinomial,
    'cat-mix-gauss' : load_gauss_cond
    
}


def main():

    load_data('mvn-test1', data_params=mvn_test1)
    load_data('mvn-test2', data_params=mvn_test2)


if __name__ == '__main__':
    main()
