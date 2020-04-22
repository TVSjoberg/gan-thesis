import numpy as np
from gan_thesis.data.datagen import r_corr, rand_prop
from itertools import chain

unif = np.random.uniform  # Shorthand
rint = np.random.randint

seed = 123
np.random.seed(seed)

mvn_test1 = {
    # 3 INDEPENDENT features
    # 1 standard normal, 1 high mean, 1 high var
    #
    'n_samples': 10000,
    'mean': [0, 3, 0],
    'var': [1, 1, 5],
    'corr': np.eye(3).tolist(),
    'seed': seed
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
             [-0.3, 0.8, 1]],
    'seed': seed
}


mvn_test3 = {
    'n_samples': 10000,
    'mean': unif(-3, 3, 9).tolist(),
    'var': unif(0, 1, 9).tolist(),
    'corr': r_corr(9).tolist(),
    'seed': seed

}
ln_test1 = mvn_test1.copy()
ln_test2 = mvn_test2.copy()
ln_test3 = mvn_test3.copy()

# feature oberoende med
# Multi modality
# 2 modes
# 2 modes minority
# 2 modes minority close
# 5 modes
#rand_prop = unif(0, 1, 5)
#rand_prop = (rand_prop/sum(rand_prop))

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
        np.eye(4).tolist() for i in range(4)
            ],

    'vars': [
        np.ones((4, 1)).tolist() for i in range(4)
            ],
    'seed': seed

}


# Komplex form m
# means mellan [0,1] Var mellan 0.5, 1.5
# proportion 2-4 slump
# 3 underlying modes
#   # 3 features
# corr mellan -0.2 - 0.2
#
#rand_prop = unif(0, 1, 3)
#rand_prop = (rand_prop/sum(rand_prop))

size = 3
mvn_mix_test2 = {
    'n_samples': 10000,
    'proportions': rand_prop(3),
    'means': [
        unif(0, 8, size).tolist() for i in range(3)

    ],
    'corrs': [
        r_corr(size).tolist() for i in range(3)
    ],
    'vars': [
        unif(0, 1, size).tolist() for i in range(3)
    ],
    'seed': seed

}


# ln COPY AV MVN

cat_test1 = {
    'n_samples': 10000,
    'probabilities': [
        [0.5, 0.5],
        [0.1, 0.2, 0.7],
        rand_prop(5).tolist()
    ],
    'seed': seed

}

cond_cat_test1 = {
    'n_samples': 10000,
    'ind_probs': [
        [1/2, 3/8, 1/8]
    ],
    'cond_probs': [
        [
            [
                [0.8, 0.1, 0.1]
            ],
            [
                [0.1, 0.8, 0.1]],
            [
                [0.1, 0.1, 0.8]
            ]
        ]
    ],
    'seed': seed
}

n_ind = 3
n_cond = [3,3,3]
n_cat = n_ind+sum(n_cond)
n_lab_ind = rint(3, 6, n_ind)
n_lab_cond = [rint(3, 6, n_cond[i]) for i in range(n_ind)]
n_labs = n_lab_ind.tolist()+list(chain(*n_lab_cond))

cond_cat_test2 = {
    'n_samples' : 10000,
    'ind_probs' : [
        rand_prop(size).tolist() for size in n_lab_ind
    ],
    'cond_probs' : [[
        [
            rand_prop(size).tolist() for size in n_lab_cond[i]
        ] for j in range(n_lab_ind[i]) ] for i in range((len(n_lab_ind)))
    ],
    'seed': seed
}


gauss_mix_cond_test1 = {
    'n_samples': 10000,
    'ind_probs': [[1/3, 1/3, 1/3]],
    'means': [
        [
            [0, 0, 0],
            [2, 2, 2],
            [4, 4, 4]
        ]
    ],
    'vars': [
        [
            unif(0.5, 1.5, 3).tolist() for i in range(3)
        ]
    ],
    'corrs': [
        [
            r_corr(3).tolist() for i in range(3)
        ]
    ],
    'seed': seed
}


# Medium dataset
# 6 categorical features 2-6 labels
# first 2 defining 2 each
# each categorical defining distribution of 2 gaussian mixtures
# All gaussians means between 0 and 5
# variances between 0.5,1.5
n_ind = 2
n_cond = [2,2]
n_cat = n_ind+sum(n_cond)
n_lab_ind = rint(2, 6, n_ind)
n_lab_cond = [rint(2, 6, n_cond[i]) for i in range(n_ind)]
n_labs = n_lab_ind.tolist()+list(chain(*n_lab_cond))
n_cont_per_cat = [2 for i in (n_labs)]



gauss_mix_cond_test2 = {
    'n_samples' : 10000,
    'ind_probs' : [
        rand_prop(size).tolist() for size in n_lab_ind
    ],
    'cond_probs' : [[
        [
            rand_prop(size).tolist() for size in n_lab_cond[i]
        ] for j in range(n_lab_ind[i]) ] for i in range((len(n_lab_ind)))
    ],
    'means' : [
        [
                unif(0,5, n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
        
    ],
    'vars' : [
        [
                unif(0.5,1.2, n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
        
    ],
    'corrs' : [
        [
                r_corr(n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
    ],
    'seed': seed
}



## Large dataset 50 features
# 3 independent categorical variables with 2 dependents each
# 2-6 labels per categorical variable
# 5 gaussians per categorical variable
#same variance structure as in medium dataset

n_ind = 3
n_cond = [3 for i in range(n_ind)]
n_cat = n_ind+sum(n_cond)
n_lab_ind = rint(2, 6, n_ind)

n_lab_cond = [rint(2, 6, n_cond[i]) for i in range(n_ind)]
n_labs = n_lab_ind.tolist()+list(chain(*n_lab_cond))
n_cont_per_cat = [5 for i in (n_labs)]


gauss_mix_cond_test3 = {
    'n_samples' : 10000,
    'ind_probs' : [
        rand_prop(size).tolist() for size in n_lab_ind
    ],
    'cond_probs' : [[
        [
            rand_prop(size).tolist() for size in n_lab_cond[i]
        ] for j in range(n_lab_ind[i]) ] for i in range((len(n_lab_ind)))
    ],
    'means' : [
        [
                unif(0,5, n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
        
    ],
    'vars' : [
        [
                unif(0.5,1.2, n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
        
    ],
    'corrs' : [
        [
                r_corr(n_cont_per_cat[i]).tolist() for j in range(n_labs[i])
        ] for i in range(n_cat)
        
    ],
    'seed': seed
}