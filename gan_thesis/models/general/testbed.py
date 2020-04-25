
import shutil
from gan_thesis.evaluation.pMSE import *
from gan_thesis.evaluation.association import plot_all_association
from gan_thesis.evaluation.machine_learning import *
from gan_thesis.evaluation.plot_marginals import *
from gan_thesis.data.load_data import load_data
import os
import pandas as pd
#import gan_thesis.models.wgan.synthesizer as gan
import gan_thesis.models.wgan.synthesizer as gan
from definitions import RESULT_DIR


def main():

    #for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1', 'ln-test2']:


    
    # for data in ['cat_mix_gauss-test3',
    #              'cond_cat-test1', 'cond_cat-test2', 'ln-test3',
    #               'mvn-test1', 'mvn-test2', 'mvn-test3',
    #                'mvn_mixture-test1', 'mvn_mixture-test2',
    #                'ln-test1', 'ln-test2', 'ln-test3', 'cat-test1']:
    
#     for i in range(3):
#         print(i)
#         data = 'cat_mix_gauss-test2'
#         params = gan.DEF_PARAMS
#         params['training_set'] = data
#         params['eval'] = None
        
#         gan.main(params, optim=False)
# #   
#     for i in range(3):
#         print(i)
#         data = 'cat_mix_gauss-test2'
#         params = gan.DEF_PARAMS
#         params['training_set'] = data
#         params['gen_num_layers'] = 3
#         params['crit_num_layers'] = 3
#         params['eval'] = None
#         if i == 2:
#             params['eval'] = 'all'
        
#         gan.main(params, optim = False)
    
    # for i in range(3):
    #     print(i)
    #     data = 'cond_cat-test1'
    #     params = gan.DEF_PARAMS
    #     params['training_set'] = data
    #     params['eval'] = None
    #     if i == 2:
    #         params['eval'] = 'all'
        
        
    #     gan.main(params, optim = False)    
    
    
    # for i in range(3):
    #     print(i)
    #     data = 'adult'
    #     params = gan.DEF_PARAMS
    #     params['training_set'] = data
        
    #     params['eval'] = None
        
    #     gan.main(params, optim = False)
    data = 'adult'
    params = gan.DEF_PARAMS
    params['training_set'] = data
    params['EPOCHS'] = 900
    params['eval'] = 'all'
    params['gen_num_layers'] = 2
    params['crit_num_layers'] = 2
    gan.main(params, optim = False)
        
    

def pmse_loop():
    output = pd.DataFrame(data=None, columns=['wgan', 'tgan', 'ctgan'])
    output = []
    for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1']:
        dataset = load_data(data)
        temp_output = []
        #for model in ['wgan', 'tgan', 'ctgan']:
        model = 'wgan'
        print('dataset: '+data+', model: '+model)
        samples = dataset.samples.get(model)
        samples_oh = pd.get_dummies(samples)
        train = dataset.train
        train_oh = pd.get_dummies(train)
        concat = df_concat_ind(train_oh, samples_oh, ind='ind')
        pmse = pMSE(concat, ind_var='ind')
        null_pmse = null_pmse_est(concat, ind_var='ind', n_iter=100)
        temp_output.append(pmse/null_pmse)
        output.append(temp_output)
    output = pd.DataFrame(data=output, columns=['wgan'], index=['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1', 'ln-test2'])
    output.to_csv(os.path.join(RESULT_DIR, 'pmse.csv'))

def ass_loop():
    for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1']:

        dataset = load_data(data)
        plot_all_marginals(dataset, data)


if __name__ == '__main__':
    main()