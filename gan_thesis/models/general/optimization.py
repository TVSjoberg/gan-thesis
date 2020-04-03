from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
import os
from gan_thesis.models.general.utils import save_json, HiddenPrints


def optimize(space, file_path=None, max_evals=5):
    if space.get('model') == 'ctgan':
        from gan_thesis.models.ctgan.synthesizer import build_and_train, sampler, optim_loss
    # elif space.get('model') == 'tgan':
        # from gan_thesis.models.tgan.synthesizer import build_and_train, sampler, optim_loss
    # elif space.get('model') == 'wgan':
        # from gan_thesis.models.wgan.synthesizer import build_and_train, sampler, optim_loss

    def objective(params):
        """Objective function for GAN Hyperparameter Tuning"""

        with HiddenPrints():  # Suppresses normal print functions
            my_gan = build_and_train(params)
        samples = sampler(my_gan, params)
        loss = optim_loss(samples, params)

        params['loss'] = loss
        # save_json(params, os.path.join(__file__, ))

        del my_gan, samples

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    # Trials object to track progress
    bayes_trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    if file_path is not None:
        save_json(best, file_path)

    return best, bayes_trials
