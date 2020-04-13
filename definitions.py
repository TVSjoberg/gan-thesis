import os
import numpy as np
from gan_thesis.data.datagen import r_corr


TEST_IDENTIFIER = ''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets', TEST_IDENTIFIER)
RESULT_DIR = os.path.join(ROOT_DIR, 'results', TEST_IDENTIFIER)


mvn_test1 = {
    # 3 INDEPENDENT features
    # 1 standard normal, 1 high mean, 1 high var
    #
    'n_samples': 10000,
    'mean': [0, 3, 0],
    'var': [1, 1, 5],
    'corr': np.eye(3).tolist()
}

mvn_test2 = {
    # medium positive
    # medium negative
    # correlations
    'n_samples': 10000,
    'mean': [0, 0, 0],
    'var': [1, 1, 1],
    'corr': [[1, 0.3, -0.3],
             [0.3, 1, 0.8],
             [-0.3, 0.8, 1]]
}

ln_test1 = mvn_test1.copy()
ln_test2 = mvn_test2.copy()

# feature oberoende med
# Multi modality
# 2 modes
# 2 modes minority
# 2 modes minority close
# 5 modes
unif = np.random.uniform
rand_prop = unif(0, 1, 5)
rand_prop = (rand_prop/sum(rand_prop))

mvn_mix_test1 = {
    'n_samples': 10000,
    'proportions':
        [0.05, 0.10, 0.35, 0.5],

    'means': [[
        0, 0, 0, 4
    ], [
        0, 4, 1.5, 0
    ], [
        0, 0, 0, 0
    ], [
        4, 0, 0, 0
    ]
    ],

    'corrs': [
        np.eye(4).tolist(),
        np.eye(4).tolist(),
        np.eye(4).tolist(),
        np.eye(4).tolist()
            ],

    'vars': [
        np.ones((4, 1)).tolist(),
        np.ones((4, 1)).tolist(),
        np.ones((4, 1)).tolist(),
        np.ones((4, 1)).tolist()
            ]

}


# Komplex form m
# means mellan [0,1] Var mellan 0.5, 1.5
# proportion 2-4 slump
# 3 underlying modes
#   # 3 features
# corr mellan -0.2 - 0.2
#
rand_prop = unif(0, 1, 3)
rand_prop = (rand_prop/sum(rand_prop))

size = 3
mvn_mix_test2 = {
    'n_samples': 10000,
    'proportions': rand_prop,
    'means': [
        unif(0, 8, size).tolist() for i in range(3)

    ],
    'corrs': [
        r_corr(size).tolist(),
        r_corr(size).tolist(),
        r_corr(size).tolist(),
    ],
    'vars': [
        unif(0, 1, size).tolist() for i in range(3)
    ]

}


# ln COPY AV MVN
