import numpy as np
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
import csv
import os
from models.ctgan.main import build_and_train, sampler, optim_loss
from models.general.utils import save_json

# Max iterations of the Bayesian Optimization algorithm
MAX_EVALS = 100

# Define the search space
space = {
    'embedding_dim': 2 ** hp.quniform('embedding_dim', 4, 9, 1),
    'gen_num_layers': hp.quniform('gen_num_layers', 1, 5, 1),
    'gen_layer_sizes': 2 ** hp.quniform('gen_layer_sizes', 4, 9, 1),
    'crit_num_layers': hp.quniform('crit_num_layers', 1, 5, 1),
    'crit_layer_sizes': 2 ** hp.quniform('crit_layer_sizes', 4, 9, 1),
    'l2scale': hp.loguniform('l2scale', np.log10(10 ** -6), np.log10(0.2)),
    'batch_size': 50 * hp.quniform('batch_size', 1, 50, 1)
}


def objective(params):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    my_ctgan = build_and_train(params)
    samples = sampler(my_ctgan)
    loss = optim_loss(samples)

    params['loss'] = loss
    save_json(params, os.path.join(__file__,))

    del my_ctgan, samples

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


def optimize(file_path):
    # Trials object to track progress
    bayes_trials = Trials()

    # File to save results
    out_file = 'optimization_trials.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration'])
    of_connection.close()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
    save_json(best, file_path)

    return best, bayes_trials
