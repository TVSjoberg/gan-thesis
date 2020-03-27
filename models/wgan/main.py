from models.wgan.data import load_credit_data

from models.wgan.wgan import *


def main():
    """params:
        output_dim: integer dimension of the output variables.
        Note that this includes the one-hot encoding of the categorical varibles

        latent_dim: integer dimension of random noise sampled for the generator

        gen_dim: tuple with the hidden layer dimension for the generator

        crit_dim tuple with hidden layer dimension for the critic

        mode: 'wgan' or 'wgan-gp', deciding which loss function to use

        gp_const: Gradient penalty constant. Only needed if mode == 'wgan-gp'

        n_critic: Number of critic learning iterations per generator iteration

        Checkpoints: yet to be added... """

    testing_params = {
        'output_dim': 16,
        'latent_dim': 128,
        'gen_dim': (256, 256),
        'crit_dim': (256, 256),
        'mode': 'wgan-gp',
        'gp_const': 10,
        'n_critic': 16
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    my_wgan = WGAN(testing_params)
    dataframe = load_credit_data()
    cat_cols = ['Gender', 'Student', 'Married', 'Ethnicity']
    int_cols = ['Cards', 'Age', 'Education']
    cont_cols = ['Income', 'Limit', 'Rating', 'Balance']
    cols_to_scale = cont_cols + int_cols
    my_wgan.train(dataframe, 100, cat_cols, cols_to_scale)


if __name__ == "__main__":
    main()
