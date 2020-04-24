import gan_thesis.models.ctgan.synthesizer as gan
# import gan_thesis.models.tgan.synthesizer as gan
import shutil
from gan_thesis.evaluation.pMSE import *
from gan_thesis.evaluation.association import plot_all_association
from gan_thesis.evaluation.machine_learning import *
from gan_thesis.evaluation.plot_marginals import *
from gan_thesis.data.load_data import load_data
import os
# import gan_thesis.models.wgan.synthesizer as gan
from definitions import RESULT_DIR


def main():

    for data in ['cat-test1',
                 'cat_mix_gauss-test1', 'cat_mix_gauss-test2', 'cat_mix_gauss-test3',
                 'cond_cat-test1', 'cond_cat-test2',
                 'ln-test2', 'ln-test3',
                 'mvn-test2', 'mvn-test3',
                 'mvn_mixture-test1', 'mvn_mixture-test2']:
        params = gan.DEF_PARAMS
        params['training_set'] = data
        gan.main(params, optim=False)


def pmse_loop():
    output = pd.DataFrame(data=None, columns=['wgan', 'tgan', 'ctgan'])
    output = []
    for data in ['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1', 'ln-test2']:
        dataset = load_data(data)
        temp_output = []
        for model in ['wgan', 'tgan', 'ctgan']:
            print('dataset: '+data+', model: '+model)
            samples = dataset.samples.get(model)
            samples_oh = pd.get_dummies(samples)
            train = dataset.train
            train_oh = pd.get_dummies(train)
            pmse = pMSE_ratio(real_df=train_oh, synth_df=samples_oh)
            temp_output.append(pmse['logreg'])
        output.append(temp_output)
    output = pd.DataFrame(data=output, columns=['wgan', 'tgan', 'ctgan'], index=['cat_mix_gauss-test1', 'cond_cat-test1', 'cat-test1', 'mvn-test1', 'mvn-test2', 'mvn-test3', 'mvn_mixture-test1', 'mvn_mixture-test2', 'ln-test1', 'ln-test2'])
    output.to_csv(os.path.join(RESULT_DIR, 'pmse.csv'))


def eval_loop():
    for data in ['cat-test1',
                 'cat_mix_gauss-test1', 'cat_mix_gauss-test2', 'cat_mix_gauss-test3',
                 'cond_cat-test1', 'cond_cat-test2',
                 'ln-test2', 'ln-test3',
                 'mvn-test2', 'mvn-test3',
                 'mvn_mixture-test1', 'mvn_mixture-test2']:
        print('Starting MLE evaluation on samples...')
        dset = load_data(data)
        plot_all_predictions_by_dimension(dataset, data)
        print('Plotting marginals of real and sample data...')
        plot_all_marginals(dataset, data)
        print('Plotting association matrices...')
        plot_all_association(dataset, data)


def ass_loop():
    for data in ['ln-test2']:

        dataset = load_data(data)
        plot_all_association(dataset, data)


if __name__ == '__main__':
    main()