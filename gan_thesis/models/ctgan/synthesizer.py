from ctgan import CTGANSynthesizer
from gan_thesis.evaluation.machine_learning import plot_predictions_by_dimension
from gan_thesis.evaluation.plot_marginals import plot_marginals
from gan_thesis.evaluation.association import plot_association
from gan_thesis.evaluation.pMSE import *
from gan_thesis.data.load_data import *
from gan_thesis.models.general.utils import save_model, load_model, save_json
from gan_thesis.models.general.optimization import optimize
import os
import pandas as pd
from definitions import RESULT_DIR
from hyperopt import hp

EPOCHS = 1

# HYPEROPT SPACE
space = {
    'embedding_dim': hp.quniform('embedding_dim', 16, 512, 2),
    'gen_num_layers': hp.quniform('gen_num_layers', 1, 5, 1),
    'gen_layer_sizes': hp.quniform('gen_layer_sizes', 16, 512, 2),
    'crit_num_layers': hp.quniform('crit_num_layers', 1, 5, 1),
    'crit_layer_sizes': hp.quniform('crit_layer_sizes', 16, 512, 2),
    'l2scale': hp.loguniform('l2scale', np.log10(10 ** -6), np.log10(0.2)),
    'batch_size': hp.quniform('batch_size', 50, 500, 50)
}


def build_and_train(params):

    gen_layers = [int(params['gen_layer_sizes'])] * int(params['gen_num_layers'])
    print(gen_layers)
    crit_layers = [int(params['crit_layer_sizes'])] * int(params['crit_num_layers'])
    print(crit_layers)
    my_ctgan = CTGANSynthesizer(embedding_dim=int(params['embedding_dim']),
                                gen_dim=gen_layers,
                                dis_dim=crit_layers,
                                batch_size=int(params['batch_size']),
                                l2scale=params['l2scale'])
    print('Fitting a CTGAN model for {0} epochs...'.format(EPOCHS))
    d = params.get('dataset')
    my_ctgan.fit(d.train, d.info.get('discrete_columns'), epochs=EPOCHS)
    print('Successfully fitted a CTGAN model')

    return my_ctgan


def sampler(my_ctgan, params):
    d = params.get('dataset')
    samples = my_ctgan.sample(len(d.train))
    col = d.train.columns
    samples.columns = col
    samples = samples.astype(d.train.dtypes)
    ctgan_dataset = Dataset(d.train, d.test, samples, d.info)

    return ctgan_dataset


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
            'l2scale': 10**-6,
            'batch_size': 500
        }

    if optim:
        params.update(space)  # Overwrite NN hyperparameters with stochastic variant from top of file

    print('Starting CTGAN main script with following parameters:')
    for key in params:
        print(key, params[key])
    params['model'] = 'ctgan'

    # Load dataset
    dataset = load_data(params.get('training_set'))
    params['dataset'] = dataset
    print('Successfully loaded dataset {0}'.format(params.get('training_set')))

    if optim:
        # Optimize or load CTGAN model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model') + '_optimized')
        if os.path.isfile(filename):
            my_ctgan = load_model(filename)
            print('Successfully loaded old optimized CTGAN model from {0}'.format(filename))
        else:
            best, trials = optimize(params, filename+'.json')
            best['dataset'] = dataset
            my_ctgan = build_and_train(best)
            save_model(my_ctgan, filename, force=True)
            print('Saved the optimized CTGAN model at {0}'.format(filename))
    else:
        # Train or load CTGAN model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model'))
        if os.path.isfile(filename):
            my_ctgan = load_model(filename)
            print('Successfully loaded old CTGAN model from {0}'.format(filename))
        else:
            my_ctgan = build_and_train(params=params)
            save_model(my_ctgan, filename, force=True)
            print('Saved the CTGAN model at {0}'.format(filename))

    # Sample from model
    print('Sampling from the CTGAN model...')
    samples = sampler(my_ctgan, params)
    save_samples(samples.data, params['training_set'], model='ctgan')
    print('Saved the CTGAN samples')

    # Evaluate fitted model
    if params['eval'] == 'all':
        print('Starting MLE evaluation on samples...')
        discrete_columns, continuous_columns = dataset.get_columns()
        plot_predictions_by_dimension(real=dataset.train, samples=samples.data, data_test=dataset.test,
                                      discrete_columns=discrete_columns, continuous_columns=continuous_columns,
                                      dataset=params.get('training_set'), model='ctgan')
        print('Plotting marginals of real and sample data...')
        plot_marginals(dataset.train, samples.data, params.get('training_set'), 'ctgan')
        print('Plotting association matrices...')
        diff = plot_association(dataset, samples, params.get('training_set'), 'ctgan')
        print(diff)
        save_json(diff, os.path.join(RESULT_DIR, params.get('training_set'), 'ctgan', 'association_difference'))


if __name__ == "__main__":
    main(params=None, optim=True)