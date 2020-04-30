from gan_thesis.evaluation.machine_learning import plot_predictions_by_dimension
from gan_thesis.evaluation.plot_marginals import plot_marginals
from gan_thesis.evaluation.association import plot_association
from gan_thesis.evaluation.pMSE import *
from gan_thesis.data.load_data import *
from gan_thesis.models.general.utils import save_model, load_model, save_json
#from gan_thesis.models.general.optimization import optimize
from gan_thesis.models.wgan.wgan import *

import datetime
import os
import pandas as pd
from definitions import RESULT_DIR
#from hyperopt import hp

EPOCHS = 300

DEF_PARAMS = {
            'eval': 'all',
            # NN Hyperparameters
            'EPOCHS' : EPOCHS,
            'embedding_dim': 128,
            'gen_num_layers': 2,
            'gen_layer_sizes': 256,
            'crit_num_layers': 2,
            'crit_layer_sizes': 256,
            'mode' : 'wgan-gp',
            'gp_const' : 10,
            'n_critic' : 5,
            'batch_size': 500,
            'hard' : False,
            'temp_anneal' : False
        }

# HYPEROPT SPACE
# space = {
#     'embedding_dim': 2 ** hp.quniform('embedding_dim', 4, 9, 1),
#     'gen_num_layers': hp.quniform('gen_num_layers', 1, 5, 1),
#     'gen_layer_sizes': 2 ** hp.quniform('gen_layer_sizes', 4, 9, 1),
#     'crit_num_layers': hp.quniform('crit_num_layers', 1, 5, 1),
#     'crit_layer_sizes': 2 ** hp.quniform('crit_layer_sizes', 4, 9, 1),
#     'l2scale': hp.loguniform('l2scale', np.log10(10 ** -6), np.log10(0.2)),
#     'batch_size': 50 * hp.quniform('batch_size', 1, 50, 1)
# }


def build_and_train(params):
    gen_layers = [int(params['gen_layer_sizes'])] * int(params['gen_num_layers'])
    crit_layers = [int(params['crit_layer_sizes'])] * int(params['crit_num_layers'])
    d = params.get('dataset')
    
    params['gen_dim'] = gen_layers
    params['crit_dim'] = crit_layers
    
    params['output_dim'] = d.info.get('dim')
    epchs = params['EPOCHS']

    my_wgan = WGAN(params)
    print('Fitting a wgan model for {0} epochs...'.format(epchs))
    max_iter = 7 ##Wgan overflows at ~950 epochs
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(epchs//max_iter):
        my_wgan.train(d.train, max_iter, 
                    d.info.get('discrete_columns'), 
                    d.info.get('continuous_columns'), 
                    batch_size=params['batch_size'], 
                    hard=params['hard'],
                    temp_anneal = params['temp_anneal'], 
                    input_time=curr_time)
    my_wgan.train(d.train, epchs%max_iter, 
                    d.info.get('discrete_columns'), 
                    d.info.get('continuous_columns'), 
                    batch_size=params['batch_size'], 
                    hard=params['hard'],
                    temp_anneal = params['temp_anneal'], 
                    input_time=curr_time)
    
    
    print('Successfully fitted a wgan model')

    return my_wgan


def sampler(my_wgan, params):
    d = params.get('dataset')
    samples = my_wgan.sample_df(len(d.train))
    #col = d.train.columns
    #samples.columns = col
    samples = samples.astype(d.train.dtypes)

    return samples


def optim_loss(samples, params):
    d = params.get('dataset')
    optim_df = add_indicator(real_df=d.train, synth_df=samples)

    # one-hot-encode discrete features
    one_hot_df = pd.get_dummies(optim_df, columns=d.info.get('discrete_columns'))

    print(one_hot_df.head())
    loss = pMSE(one_hot_df)
    print(loss)

    return loss


def main(params=None, optim=False):
    if params is None:
        params = {
            # Regular parameters
            'training_set': 'cat_mix_gauss-test1',
            'eval': 'all',
            # NN Hyperparameters
            'EPOCHS' : EPOCHS,
            'embedding_dim': 128,
            'gen_num_layers': 2,
            'gen_layer_sizes': 256,
            'crit_num_layers': 2,
            'crit_layer_sizes': 256,
            'mode' : 'wgan-gp',
            'gp_const' : 10,
            'n_critic' : 5,
            'batch_size': 500,
            'hard' : False,
            'temp_anneal' : False
        }

    if optim:
        params.update(space)  # Overwrite NN hyperparameters with stochastic variant from top of file

    print('Starting wgan-gp main script with following parameters:')
    for key in params:
        print(key, params[key])
    params['model'] = 'wgan'

    # Load dataset
    print(params.get('training_set'))
    dataset = load_data(params.get('training_set'))
    params['dataset'] = dataset
    
    print('Successfully loaded dataset {0}'.format(params.get('training_set')))

    alist = params.get('training_set').split(sep='-', maxsplit=1)
    
    basepath = os.path.join(RESULT_DIR, *alist, params.get('model'))
    filepath = os.path.join(basepath, '{0}_{1}_ass_diff.json'.format(alist[0], params.get('model')))
    if params.get('log_directory') != None:
        params['log_directory'] = os.path.join(basepath,params['log_directory'])
    else:
        params['log_directory'] = basepath
    
    if optim:
        # Optimize or load wgan model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model') + '_optimized')
        if os.path.isfile(filename):
            my_wgan = load_model(filename)
            print('Successfully loaded old optimized wgan model from {0}'.format(filename))
        else:
            best, trials = optimize(params, filename+'.json')
            my_wgan = build_and_train(best)
            save_model(my_wgan, filename, force=True)
            print('Saved the optimized wgan model at {0}'.format(filename))
    else:
        # Train or load wgan model
        filename = os.path.join(RESULT_DIR, params.get('training_set'), params.get('model'))

        my_wgan = build_and_train(params=params)
        # try:
        #     save_model(my_wgan, filename, force = True)
        #     print('Saved the wgan model at {0}'.format(filename))
        # except Exception as e:
        #     print('Model was not saved due to an error: {0}'.format(e))
        #     #os.remove(filename)
            
        #save_model(my_wgan, filename, force=True)
        #print('Saved the wgan model at {0}'.format(filename))

    # Sample from model
    print('Sampling from the wgan model...')
    samples = sampler(my_wgan, params)
    save_samples(samples, params['training_set'], model='wgan')
    print('Saved the wgan samples')

    # Evaluate fitted model
    if params['eval'] == 'all':
        print('Starting MLE evaluation on samples...')
        discrete_columns, continuous_columns = dataset.get_columns()
        plot_predictions_by_dimension(real=dataset.train, samples=samples, data_test=dataset.test,
                                       discrete_columns=discrete_columns, continuous_columns=continuous_columns,
                                       dataset=params.get('training_set'), model='wgan')
        print('Plotting marginals of real and sample data...')
        plot_marginals(dataset.train, samples, params.get('training_set'), 'wgan')
        print('Plotting association matrices...')
        diff = plot_association(dataset, samples, params.get('training_set'), params.get('model'))
        print(diff)
        save_json(diff, filepath)


if __name__ == "__main__":
    main()
