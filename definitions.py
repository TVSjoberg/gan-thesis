import os
import numpy as np


TEST_IDENTIFIER = ''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets', TEST_IDENTIFIER)
RESULT_DIR = os.path.join(ROOT_DIR, 'results', TEST_IDENTIFIER)

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


## feature oberoende med
# Multi modality
# 2 modes
# 2 modes minority
# 2 modes minority
# 5 modes
rand_prop = np.random.uniform(0,1,5)
rand_prop= (rand_prop/sum(rand_prop))

mvn_mix_test1 = {
    'n_samples' : 10000,
    'proportions' : [
        [0.5, 0.5],
        [0.1, 0.9],
        [0.01, 0.9],
        [0.1, 0.9],
        rand_prop
        ],
    'means' : [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0.5],
        np.random.uniform(-2,2,size = 5)
        ],
    'corrs' : [
        np.eye(5)
    ]

}

# Komplex form m
    # means mellan [0,1] Var mellan 0.5, 1.5
    # proportion 2-4 slump
    # 3 underlying modes
#   # 3 features
    # corr mellan -0.2 - 0.2
#

# ln COPY AV MVN


