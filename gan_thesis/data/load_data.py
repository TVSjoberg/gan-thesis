import pandas as pd
import os
import json
import shutil
import numpy as np
from gan_thesis.data.datagen import *
from definitions import DATA_DIR


def load_data(dataset, data_params=None):
    pathname = os.path.join(DATA_DIR, dataset)
    filelist = ['train.csv', 'test.csv', 'data.csv', 'info.json']
    filelist = map(lambda x: os.path.join(pathname, x), filelist)
    if not all([os.path.isfile(f) for f in filelist]):
        if os.path.exists(pathname):
            shutil.rmtree(pathname)
        os.mkdir(pathname)
        load_wrapper[dataset](pathname, data_params)

    train = pd.read_csv(os.path.join(pathname, 'train.csv'))
    test = pd.read_csv(os.path.join(pathname, 'test.csv'))
    df = pd.read_csv(os.path.join(pathname, 'data.csv'))
    with open(os.path.join(pathname, 'info.json'), "r") as read_file:
        info = json.load(read_file)

    return train, test, df, info


def load_adult(dirname, *args):
    n_test = 9600  # Same as CTGAN paper
    info = {
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
                               'education.num',
                               'capital-gain',
                               'capital-loss',
                               'hours-per-week'],
        "n_test": n_test
    }

    cc = info.get('columns')
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                     names=cc, header=0)

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
    covs = data_params['covs']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = mixture_gauss(n_samples, proportions, means, covs, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []

    save_data(df, info, pathname, True)


def load_mvn(pathname, data_params):
    n_samples = data_params['n_samples']
    mean = data_params['mean']
    cov = data_params['cov']
    if data_params.get('seed') is None:
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')

    df, info = multivariate_df(n_samples, mean, cov, seed)
    info['seed'] = seed
    info['continuous_columns'] = df.columns.to_list()
    info['discrete_columns'] = []
    save_data(df, info, pathname, True)

def load_ln_mixture(pathname, data_params):
    n_samples = data_params['n_samples']
    proportions = data_params['proportions']
    means = data_params['means']
    covs = data_params['covs']
    if (data_params.get('seed') == None):
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')
        
    df, info = mixture_log_normal(n_samples, proportions, means, covs, seed)
    info['seed'] = seed
    
    save_data(df, info, pathname, True)
    
def load_ln(pathname, data_params):
    n_samples = data_params['n_samples']
    mean = data_params['mean']
    cov = data_params['cov']
    if (data_params.get('seed') == None):
        seed = np.random.randint(10000)
    else:
        seed = data_params.get('seed')
        
    df, info = log_normal_df(n_samples, mean, cov, seed)
    info['seed'] = seed
    
    save_data(df, info, pathname, True)

    


def save_data(df, info, dirname):
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


def save_samples(df, dataset, model, force=False):
    rootname = os.path.dirname(__file__)
    pathname = os.path.join(rootname, dataset)
    filename = os.path.join(pathname, model + '_samples.csv')
    if os.path.isfile(filename) and not force:
        return

    df.to_csv(filename, index=False)


load_wrapper = {
    'adult' : load_adult,
    'mvn'   : load_mvn,
    'mvn-mixture' : load_mvn_mixture,
    'ln' : load_ln,
    'ln-mixture' : load_ln_mixture
}


def main():
    ln_params = {
        'n_samples' : 10000,
        'mean'      : [0,0.5,1],
        'cov'       : (np.eye(3)+0.2).tolist()
    }
    ln_mix_params = {
        'n_samples' : 10000,
        'proportions' : [0.5, 0.5],
        'means'      : [[0,0.5,1], [2,3,5]],
        'covs'       : [(np.eye(3)+0.2).tolist(),(np.eye(3)*3+1).tolist()]
    }
    load_data('ln', ln_params)
    load_data('ln-mixture', ln_mix_params)


if __name__ == '__main__':
    main()
