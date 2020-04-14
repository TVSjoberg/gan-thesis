
import shutil
import os
import gan_thesis.models.wgan.synthesizer as gan



<<<<<<< HEAD
def main():
=======
    for data in ['mvn-test2_highfeatures']:
>>>>>>> refs/remotes/origin/master

    #for data in ['mvn-test1', 'mvn-test2', 'mvn_mixture-test1', 'mvn_mixture-test2']:
    for data in  ['cond_cat-test1', 'cat_mix_gauss-test1']:
        params = gan.DEF_PARAMS
        params['training_set'] = data
        gan.main(params, optim=False)


if __name__ == '__main__':
    main()
