from tgan.model import TGANModel
from gan_thesis.evaluation.machine_learning import plot_predictions_by_dimension
from gan_thesis.evaluation.plot_marginals import plot_marginals
from gan_thesis.evaluation.association import plot_association
from gan_thesis.evaluation.pMSE import *
from gan_thesis.data.load_data import *
from gan_thesis.models.general.utils import save_json
from gan_thesis.models.general.optimization import optimize
import os
import pandas as pd
from definitions import RESULT_DIR
from hyperopt import hp
import tensorflow as tf

EPOCHS = 1
# HYPEROPT SPACE
space = {
    'embedding_dim': hp.quniform('embedding_dim', 16, 512, 2),
    'gen_num_layers': hp.quniform('gen_num_layers', 100, 600, 100),
    'gen_layer_sizes': hp.quniform('gen_layer_sizes', 100, 600, 100),
    'crit_num_layers': hp.quniform('crit_num_layers', 1, 5, 1),
    'crit_layer_sizes': hp.quniform('crit_layer_sizes', 100, 600, 100),
    'learning_rate': hp.loguniform('l2scale', np.log10(10 ** -6), np.log10(0.2)),
    'batch_size': hp.quniform('batch_size', 50, 500, 50)
}

# DEFAULT PARAMS
DEF_PARAMS = {
    # Regular parameters
    'eval': 'all',
    # NN Hyperparameters
    'embedding_dim': 200,
    'gen_num_layers': 100,
    'gen_layer_sizes': 100,
    'crit_num_layers': 1,
    'crit_layer_sizes': 100,
    'l2scale': 10 ** -6,
    'l2norm': 10 ** -5,
    'learning_rate': 10 ** -3,
    'batch_size': 200
}


def build_and_train(params):
    tf.reset_default_graph()
    gen_layers = [int(params['gen_layer_sizes'])] * int(params['gen_num_layers'])
    print(gen_layers)
    crit_layers = [int(params['crit_layer_sizes'])] * int(params['crit_num_layers'])
    print(crit_layers)
    d = params.get('dataset')
    continuous_columns = d.info.get('continuous_columns')
    print('Batch Size:' + str(params.get('batch_size')))
    savestr = str(np.random.randint(1, 999999))
    my_tgan = TGANModel(continuous_columns=continuous_columns, batch_size=int(params.get('batch_size')),
                        z_dim=int(params.get('embedding_dim')), learning_rate=params.get('learning_rate'),
                        num_gen_rnn=int(params.get('gen_num_layers')), num_gen_feature=int(params.get('gen_layer_sizes')),
                        num_dis_layers=int(params.get('crit_num_layers')), num_dis_hidden=int(params.get('crit_layer_sizes')),
                        max_epoch=EPOCHS, steps_per_epoch=20,
                        restore_session=False, output=savestr)
    print('Fitting a TGAN model for {0} epochs...'.format(EPOCHS))
    train_copy = d.train.copy()
    my_tgan.fit(train_copy)
    print('Successfully fitted a TGAN model')

    return my_tgan


def sampler(my_tgan, params):
    d = params.get('dataset')
    train = d.train
    samples = my_tgan.sample(len(train))
    col = train.columns.to_list()
    samples.columns = col
    print(train.head())
    samples = samples.astype(train.dtypes)
    tgan_dataset = Dataset(d.train, d.test, samples, d.info)

    return tgan_dataset


def optim_loss(samples, params):
    ind = 'ind'
    d = params.get('dataset')
    optim_df = df_concat_ind(real_df=d.train, gen_df=samples, ind=ind)

    # one-hot-encode discrete features
    one_hot_df = pd.get_dummies(optim_df, columns=d.info.get('discrete_columns'))

    print(one_hot_df.head())
    loss = pMSE(one_hot_df, ind_var=ind)
    print(loss)

    return loss


def main(params=None, optim=True):
    if params is None:
        params = {
            # Regular parameters
            'training_set': 'ln',
            'eval': 'all',
            # NN Hyperparameters
            'embedding_dim': 128,
            'gen_num_layers': 2,
            'gen_layer_sizes': 256,
            'crit_num_layers': 2,
            'crit_layer_sizes': 256,
            'learning_rate': 10**-6,
            'batch_size': 500,
            'training_iter': 1
        }

    if optim:
        params.update(space)  # Overwrite NN hyperparameters with stochastic variant from top of file

    print('Starting TGAN main script with following parameters:')
    for key in params:
        print(key, params[key])
    params['model'] = 'tgan'

    # Load dataset
    dataset = load_data(params.get('training_set'))
    params['dataset'] = dataset
    print('Successfully loaded dataset {0}'.format(params.get('training_set')))

    if optim:
        # Optimize or load TGAN model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model') + '_optimized')
        if os.path.isfile(filename):
            my_tgan = TGANModel.load(filename)
            print('Successfully loaded old optimized TGAN model from {0}'.format(filename))
        else:
            best, trials = optimize(params, filename+'.json')
            best['dataset'] = dataset
            my_tgan = build_and_train(best)
            my_tgan.save(filename)
            print('Saved the optimized TGAN model at {0}'.format(filename))
    else:
        # Train or load CTGAN model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model') + '_default')
        if os.path.isfile(filename):
            # my_tgan = TGANModel.load(filename)
            print('Successfully loaded old TGAN model from {0}'.format(filename))
        else:
            my_tgan = build_and_train(params=params)
            # my_tgan.save(filename)
            print('Saved the TGAN model at {0}'.format(filename))

    # Sample from model
    print('Sampling from the TGAN model...')
    samples = sampler(my_tgan, params)
    save_samples(samples.data, params['training_set'], model=params.get('model'), force=True)
    print('Saved the TGAN samples')

    # Evaluate fitted model
    if params['eval'] == 'all':
        print('Starting MLE evaluation on samples...')
        discrete_columns, continuous_columns = dataset.get_columns()
        plot_predictions_by_dimension(real=dataset.train, samples=samples.data, data_test=dataset.test,
                                      discrete_columns=discrete_columns, continuous_columns=continuous_columns,
                                      dataset=params.get('training_set'), model=params.get('model'))
        print('Plotting marginals of real and sample data...')
        plot_marginals(dataset.train, samples.data, params.get('training_set'), params.get('model'))
        print('Plotting association matrices...')
        diff = plot_association(dataset, samples, params.get('training_set'), params.get('model'))
        print(diff)
        save_json(diff, os.path.join(RESULT_DIR, params.get('training_set'), params.get('model'), 'association_difference'))


if __name__ == "__main__":
    main(params=None, optim=True)
