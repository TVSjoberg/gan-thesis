import tensorflow as tf
import pandas as pd
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
import time

from utils import *
from data import *

'''
gå från dataframe till dataset, m  col_names/nums
len(data['name'].unique())

'''

class wgan:

    def __init__(self, params):
        '''params:
        output_dim: integer dimension of the output variables.
        Note that this includes the one-hot encoding of the categorical varibles
        
        latent_dim: integer dimension of random noise sampled for the generator

        gen_dim: tuple with the hidden layer dimension for the generator

        crit_dim tuple with hidden layer dimension for the critic

        mode: 'wgan' or 'wgan-gp', deciding which loss function to use

        gp_const: Gradient penalty constant. Only needed if mode == 'wgan-gp'

        n_critic: Number of critic learning iterations per generator iteration

        Checkpoints: yet to be added... '''

        self.output_dim = params['output_dim']
        self.latent_dim = params['latent_dim']
        self.mode = params['mode']
        self.n_critic = params['n_critic']
        if self.mode == 'wgan-gp':  
            self.gp_const = params['gp_const']
        
        gen_dim = params['gen_dim']
        crit_dim = params['crit_dim']
        self.generator = self.makeGenerator(gen_dim)
        self.critic = self.makeCritic(crit_dim)
        self.cat_dims = ()

        self.gen_opt, self.crit_opt = self.get_opts()

        self.temperature = 0.2



    def makeGenerator(self, gen_dim):
        inputs = keras.Input(shape=(self.latent_dim,))

        if type(gen_dim) == int:
            temp_layer = layers.Dense(gen_dim, 
                    kernel_initializer = 'normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)
    
        else:
            temp_layer = layers.Dense(gen_dim[0], 
                    kernel_initializer = 'normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)
            
            for shape in gen_dim[1:]:
                temp_layer = layers.Dense(shape, 
                        kernel_initializer = 'normal')(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.ReLU()(temp_layer)
        
        
        outputs = layers.Dense(self.output_dim)(temp_layer)
        model = keras.Model(inputs=inputs,outputs=outputs)
        return model


    def makeCritic(self, crit_dim):

        inputs = keras.Input(shape=(self.output_dim,))
        if self.mode == 'wgan':
            constraint = ClipConstraint(0.01)
        else:
            constraint = None
        
        if type(crit_dim) == int:
            temp_layer = layers.Dense(crit_dim, 
                    kernel_constraint = constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)
        
        else:
            temp_layer = layers.Dense(crit_dim[0], 
                    kernel_constraint = constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)
        
            for shape in crit_dim[1:]:
                temp_layer = layers.Dense(shape, 
                        kernel_constraint = constraint)(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.LeakyReLU()(temp_layer)
            
        outputs = layers.Dense(1)(temp_layer)
        model = keras.Model(inputs=inputs,outputs=outputs)
        return model

        
    def get_opts(self):
        if self.mode == 'wgan':
            gen_opt = keras.optimizers.RMSprop(1e-5)
            crit_opt = keras.optimizers.RMSprop(1e-5)
        elif self.mode == 'wgan-gp':
            gen_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1 = 0.5, beta_2 = 0.9)
            crit_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1 = 0.5, beta_2 = 0.9)
        return gen_opt, crit_opt


    def sample_df(self, n, temperature = 0.2, hard = True, scaled = False):
        array_sample = sample(n, temperature, hard).numpy()
        df_sample = pd.DataFrame(array_sample, columns=self.oht_shuff_cols)
        
        if not scaled:
            df_sample = self.scaler.inverse_transfrom(df_sample)
            df_sample = df_sample[self.orignal_order_cols]
        
        return df_sample


    def sample(self, n, temperature = 0.2, hard = True):
        noise = tf.random.normal((n, self.latent_dim))
        sample = self.generator(noise, training = False)
        
        if self.cat_dims != ():
            sample = sample_gumbel(sample, temperature, self.cat_dims, hard)
        
        return sample
    
    

    def train(self, Dataframe, epochs, cat_cols = [], cont_cols = [], hard = True, batch_size = 32, shuffle = True):
        df = Dataframe.copy()
        self.cat_cols = cat_cols
        self.orignal_order_cols = list(df.columns) ## For restoring original order of data
        temp_li = []
        
        for cat in cat_cols:
            temp_li.append(len(df[cat].unique()))
        self.cat_dims = tuple(temp_li)

        df = data_reorder(df, self.cat_cols)
        
        self.scaler = dataScaler()
        df = self.scaler.transform(df, cont_cols, self.cat_cols)
        df = df.astype('float32')
        self.oht_shuff_cols = list(df.columns)

        dataset = df_to_dataset(df, shuffle, batch_size)
        self.train_ds(dataset, epochs, self.cat_dims, hard)





    
    def train_ds(self, dataset, epochs, cat_dims = (), hard = True):

        self.cat_dims = cat_dims

        for epoch in range(epochs):
            start = time.time()
            
            for data_batch in dataset:
                c_loss = 0
                g_loss = 0
                for i in np.arange(0,self.n_critic):
                    c_loss = self.train_step_c(data_batch, hard)
                c_loss /= self.n_critic
                g_loss = self.train_step_g(len(data_batch), hard)
            if (epoch + 1) % 5 == 0:
                #checkpoint.save(file_prefix = checkpoint_prefix)
            
                print('Time for epoch {} is {} sec \n with critic loss: {} and generator loss {}'.format(epoch+1, time.time()-start,c_loss, g_loss))


    @tf.function
    def train_step_c(self, data_batch, hard):
        tot_dim = data_batch.shape[1]
        start_cat_dim = tot_dim-sum(self.cat_dims)
        noise = tf.random.normal((len(data_batch), self.latent_dim))
        
        with tf.GradientTape() as crit_tape:

            fake_data = self.generator(noise, training = True)
            if self.cat_dims != ():
                fake_data = sample_gumbel(fake_data, self.temperature, self.cat_dims, hard)

            real_output = self.critic(data_batch, training= True)
            fake_output = self.critic(fake_data, training = True)

            crit_loss = critic_loss(real_output, fake_output)
            if self.mode == 'wgan-gp':
                gp_loss = self.gp_const*gradient_penalty(partial(self.critic), data_batch, fake_data)
                crit_loss += gp_loss

            critic_gradients = crit_tape.gradient(crit_loss, self.critic.trainable_variables)

            self.crit_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        return crit_loss

    @tf.function
    def train_step_g(self, batch_size, hard):
        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            gen_tape.watch(fake_data)
            
            if self.cat_dims != ():
                fake_data = sample_gumbel(fake_data, self.temperature, self.cat_dims, hard)
            
        
            fake_output = self.critic(fake_data, training= True)
            gen_loss = generator_loss(fake_output)

            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_opt.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        return gen_loss

    def set_temperature(self, temperature):
        self.temperature = temperature

