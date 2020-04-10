# import gan_thesis.models.ctgan.synthesizer as gan
import gan_thesis.models.tgan.synthesizer as gan
import shutil
import os
# import gan_thesis.models.wgan.synthesizer as gan


def main():

    for data in ['mvn-test1', 'mvn-test2']:

        params = gan.DEF_PARAMS
        params['training_set'] = data
        gan.main(params, optim=False)


if __name__ == '__main__':
    main()
