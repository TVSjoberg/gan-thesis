import numpy as np
from scipy.stats import random_correlation

mvn_test1 = {
    # 3 INDEPENDENT features
    # 1 standard normal, 1 high mean, 1 high var
    #
    'n_samples' : 10000,
    'mean' : [0 ,3, 0],
    'var' : [1, 1, 5],
    'corr' : np.eye(3).tolist()
}

mvn_test2 = {
    #medium positive
    #medium negative
    #correlations
    'n_samples' : 10000,
    'mean' : [0, 0, 0],
    'var' : [1, 1, 1],
    'corr' : [[1, 0.3, -0.3],
              [0.3, 1, 0.8],
              [-0.3, 0.8, 1]]
}

mvn_test1_highfeature = {
    #medium positive
    #medium negative
    #correlations
    'n_samples' : 10000,
    'mean' : np.zeros(shape=(1,10)),
    'var' : np.ones(shape=(1,10)),
    'corr' : [[1, 0.3, -0.3],
              [0.3, 1, 0.8],
              [-0.3, 0.8, 1]]
}


def mvn_test1_highfeature():
    mean = np.zeros(9)
    var = np.ones(9)
    corr = corr_matrix(9)
    return {
        'n_samples': 20000,
        'mean': mean.tolist(),
        'var': var.tolist(),
        'corr': corr.tolist()
    }

def mvn_test2_highfeature():
    mean = np.random.uniform(-3, 3, 9)
    var = np.random.uniform(1, 3, 9)
    corr = corr_matrix(9)
    return {
        'n_samples': 20000,
        'mean': mean.tolist(),
        'var': var.tolist(),
        'corr': corr.tolist()
    }


def corr_matrix(n):
    eigs = np.random.uniform(size=n)
    eigs = eigs * n / sum(eigs)
    C = random_correlation.rvs(eigs)
    return C
