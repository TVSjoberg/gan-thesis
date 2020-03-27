from ctgan import CTGANSynthesizer
from data.load_data import *
from models.general.utils import save, load
import os
import pandas as pd
from evaluation.machine_learning import plot_predictions_by_dimension
from evaluation.plot_marginals import plot_marginals


def main():
    params = {
        'training_set': 'adult',
        'output_dim': 16,
        'latent_dim': 128,
        'gen_dim': (256, 256),
        'crit_dim': (256, 256),
        'batch_size': 500,
        'epochs': 5,
        'gp_const': 10,
        'n_critic': 16,
        'eval': 'all'
    }
    print('Starting CTGAN main script with following parameters:')
    for key in params:
        print(key, params[key])

    # Load dataset 'training_set'
    train, test, df, info = load_data(params['training_set'])
    discrete_columns = info['discrete_columns']
    continuous_columns = info['continuous_columns']
    rootname = os.path.dirname(__file__)
    filename = os.path.join(rootname, 'savefiles', params['training_set'])
    print('Successfully loaded dataset {0}'.format(params['training_set']))

    # Train or load CTGAN model
    if os.path.isfile(filename):
        my_ctgan = load(filename)
        print('Successfully loaded old CTGAN model from {0}'.format(filename))
    else:
        my_ctgan = CTGANSynthesizer(embedding_dim=params['latent_dim'],
                                    gen_dim=params['gen_dim'],
                                    dis_dim=params['crit_dim'],
                                    batch_size=params['batch_size'])
        print('Fitting a CTGAN model for {0} epochs...'.format(params['epochs']))
        my_ctgan.fit(train, discrete_columns, epochs=params['epochs'])
        print('Successfully fitted a CTGAN model')
        save(my_ctgan, filename, force=True)
        print('Saved the CTGAN model at {0}'.format(filename))

    # Sample from model
    print('Sampling from the CTGAN model...')
    samples = my_ctgan.sample(len(train))
    col = train.columns
    samples.columns = col
    samples = samples.astype(train.dtypes)
    save_samples(samples, params['training_set'], model='ctgan')
    print('Saved the CTGAN samples')

    # Evaluate fitted model
    if params['eval'] == 'all':
        print('Starting MLE evaluation on samples...')
        plot_predictions_by_dimension(train, samples, test, discrete_columns, continuous_columns,
                                      identifier=os.path.join(params['training_set'], 'ctgan_ml_efficiency'))
        print('Plotting marginals of real and sample data...')
        plot_marginals(train, samples, identifier=os.path.join(params['training_set'], 'ctgan'))


if __name__ == "__main__":
    main()
