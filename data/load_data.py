import pandas as pd
import os
import json
import shutil


def load_data(dataset):
    rootname = os.path.dirname(__file__)
    pathname = os.path.join(rootname, dataset)
    filelist = ['train.csv', 'test.csv', 'data.csv', 'info.json']
    filelist = map(lambda x: os.path.join(pathname, x), filelist)
    if not all([os.path.isfile(f) for f in filelist]):
        if os.path.exists(pathname):
            shutil.rmtree(pathname)
        os.mkdir(pathname)
        load_adult(pathname)

    train = pd.read_csv(os.path.join(pathname, 'train.csv'))
    test = pd.read_csv(os.path.join(pathname, 'test.csv'))
    df = pd.read_csv(os.path.join(pathname, 'data.csv'))
    with open(os.path.join(pathname, 'info.json'), "r") as read_file:
        info = json.load(read_file)

    return train, test, df, info


def load_adult(dirname):
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
                               'education.num',
                               'capital-gain',
                               'capital-loss',
                               'hours-per-week']
    }

    cc = info.get('columns')
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                     names=cc, header=0)

    train, test = train_test_split(df=df, n_test=9600)
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
    filename = os.path.join(pathname, model+'_samples.csv')
    if os.path.isfile(filename) and not force:
        return

    df.to_csv(filename, index=False)


def main():
    load_data('adult')


if __name__ == '__main__':
    main()
