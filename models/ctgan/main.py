from ctgan import CTGANSynthesizer
from data.load_data import *
from models.general.utils import save, load
import os
import pandas as pd
from evaluation.machine_learning import plot_predictions_by_dimension
from evaluation.plot_marginals import plot_marginals
from evaluation.pMSE import *

EPOCHS = 1
TRAINING_SET = 'adult'

# Load dataset 'training_set'
train, test, df, info = load_data(TRAINING_SET)
discrete_columns = info['discrete_columns']
continuous_columns = info['continuous_columns']
rootname = os.path.dirname(__file__)
filename = os.path.join(rootname, 'savefiles', TRAINING_SET)
train = train.iloc[1:4000, :]


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
    my_ctgan.fit(train, discrete_columns, epochs=EPOCHS)
    print('Successfully fitted a CTGAN model')

    return my_ctgan


def sampler(my_ctgan):
    samples = my_ctgan.sample(len(train))
    col = train.columns
    samples.columns = col
    samples = samples.astype(train.dtypes)

    return samples


def optim_loss(samples):
    ind = 'ind'

    optim_df = df_concat_ind(real_df=train, gen_df=samples, ind=ind)

    # one-hot-encode discrete features
    one_hot_df = pd.get_dummies(optim_df, columns=discrete_columns)

    print(one_hot_df.head())
    loss = pMSE(one_hot_df, ind_var=ind)
    print(loss)

    return loss


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


    print('Successfully loaded dataset {0}'.format(params['training_set']))

    # Train or load CTGAN model
    if os.path.isfile(filename):
        my_ctgan = load(filename)
        print('Successfully loaded old CTGAN model from {0}'.format(filename))
    else:
        my_ctgan = build_and_train(params=params)
        save(my_ctgan, filename, force=True)
        print('Saved the CTGAN model at {0}'.format(filename))

    # Sample from model
    print('Sampling from the CTGAN model...')
    samples = sampler(my_ctgan)
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
