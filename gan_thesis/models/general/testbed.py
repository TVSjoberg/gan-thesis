
import shutil
from gan_thesis.evaluation.pMSE import *
from gan_thesis.evaluation.association import plot_all_association
from gan_thesis.data.load_data import load_data
import os
import gan_thesis.models.wgan.synthesizer as gan
from definitions import RESULT_DIR


def main():

    #for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1', 'ln-test2']:


    #for data in ['mvn-test1', 'mvn-test2', 'mvn_mixture-test1', 'mvn_mixture-test2']
    for data in  ['cond_cat-test1', 'cat_mix_gauss-test1', 'adult', 'cat-test1']:
        params = gan.DEF_PARAMS
        params['training_set'] = data
        gan.main(params, optim=False)
#       break

def pmse_loop():
    output = pd.DataFrame(data=None, columns=['wgan', 'tgan', 'ctgan'])
    output = []
    for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1']:
        dataset = load_data(data)
        print(dataset)
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
    for data in ['mvn-test2']:

        dataset = load_data(data)
        plot_all_association(dataset, data)
        break

if __name__ == '__main__':
    main()