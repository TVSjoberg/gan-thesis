import os
import pickle
import time
from functools import partial
import numpy as np
import datetime


import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean

from gan_thesis.models.wgan.utils import *
from gan_thesis.models.wgan.data import *



class WGAN:

    def __init__(self, params):
        """Main WGAN Model

        Args: Dictionary with
            output_dim:
                Integer dimension of the output variables including
                the one-hot encoding of the categorical variables
            embedding_dim:
                Integer dimension of random noise sampled for the generator
            gen_dim:
                Tuple with the hidden layer dimension for the generator
            crit_dim:
                Tuple with hidden layer dimension for the critic
            mode:
                'wgan' or 'wgan-gp', deciding which loss function to use
            gp_const:
                Gradient penalty constant. Only needed if mode == 'wgan-gp'
            n_critic:
                Number of critic learning iterations per generator iteration
            log_directory:
                Directory of tensorboard logs


        Checkpoints: yet to be added...
        """
        
        self.epoch_trained = 0
        self.output_dim = params['output_dim']
        self.latent_dim = params['embedding_dim']
        self.mode = params['mode']
        self.n_critic = params['n_critic']
        if self.mode == 'wgan-gp':
            self.gp_const = params['gp_const']

        gen_dim = params['gen_dim']
        crit_dim = params['crit_dim']
        self.generator = self.make_generator(gen_dim)
        self.critic = self.make_critic(crit_dim)
        self.cat_dims = ()

        self.gen_opt, self.crit_opt = self.get_opts()
        self.log_dir = params['log_directory']
        self.temperature = 0.2

    def make_generator(self, gen_dim):
        inputs = keras.Input(shape=(self.latent_dim,))

        if type(gen_dim) == int:
            temp_layer = layers.Dense(gen_dim,
                                      kernel_initializer='normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)

        else:
            temp_layer = layers.Dense(gen_dim[0],
                                      kernel_initializer='normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)

            for shape in gen_dim[1:]:
                temp_layer = layers.Dense(shape,
                                          kernel_initializer='normal')(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.ReLU()(temp_layer)

        outputs = layers.Dense(self.output_dim)(temp_layer)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def make_critic(self, crit_dim):

        inputs = keras.Input(shape=(self.output_dim,))
        if self.mode == 'wgan':
            constraint = ClipConstraint(0.01)
        else:
            constraint = None

        if type(crit_dim) == int:
            temp_layer = layers.Dense(crit_dim,
                                      kernel_constraint=constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)

        else:
            temp_layer = layers.Dense(crit_dim[0],
                                      kernel_constraint=constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)

            for shape in crit_dim[1:]:
                temp_layer = layers.Dense(shape,
                                          kernel_constraint=constraint)(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.LeakyReLU()(temp_layer)

        outputs = layers.Dense(1)(temp_layer)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_opts(self):
        if self.mode == 'wgan':
            gen_opt = keras.optimizers.RMSprop(1e-5)
            crit_opt = keras.optimizers.RMSprop(1e-5)
        elif self.mode == 'wgan-gp':
            gen_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
            crit_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        return gen_opt, crit_opt

    def sample_df(self, n, temperature=0.2, hard=True, scaled=False):
        array_sample = self.sample(n, temperature, hard).numpy()
        df_sample = pd.DataFrame(array_sample, columns=self.oht_shuff_cols)

        if not scaled:
            df_sample = self.scaler.inverse_transfrom(df_sample)
            df_sample = df_sample[self.orignal_order_cols]

        return df_sample

    def sample(self, n, temperature=0.2, hard=True):
        noise = tf.random.normal((n, self.latent_dim))
        sample = self.generator(noise, training=False)

        if self.cat_dims != ():
            sample = sample_gumbel(sample, temperature, self.cat_dims, hard)

        return sample



    def train(self, dataframe, epochs, cat_cols=None, cont_cols=None, hard=False, temp_anneal = False , batch_size=32, shuffle=True, input_time = False):
        
        
        if cont_cols is None:
            cont_cols = []
        if cat_cols is None:
            cat_cols = []
        df = dataframe.copy()
        self.cat_cols = cat_cols
        self.orignal_order_cols = list(df.columns)  # For restoring original order of data
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
        self.train_ds(dataset, epochs, len(df), batch_size, self.cat_dims, hard, temp_anneal, input_time)
        self.epoch_trained += epochs
        

    def train_ds(self, dataset, epochs, n_data, batch_size=32, cat_dims=(), hard=False, temp_anneal = False, input_time = False):

        self.cat_dims = cat_dims
        temp_increment = self.temperature/epochs # for temperature annealing
        
        self.g_loss = Mean('generator_loss', dtype = tf.float64)
        self.c_loss = Mean('critic_loss', dtype = tf.float64)
        
        
        current_time = input_time if input_time else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        generator_log_dir = self.log_dir+'\\logs\\'+current_time+'\\gradient_tape\\generator'
        critic_log_dir = self.log_dir+ '\\logs\\'+current_time+'\\gradient_tape\\critic'
        generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
        critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)
        
        
        for epoch in range(self.epoch_trained,self.epoch_trained+epochs):
            start = time.time()
            g_loss = 0
            c_loss = 0
            counter = 0
            #trace = True # Tensorboard tracing, currently not working
            for data_batch in dataset:
                # if trace:
                #     tf.summary.trace_on(graph = True, profiler = True)
                
                c_loss = self.train_step_c(data_batch, hard)
                
                # if trace:
                #     with critic_summary_writer.as_default():
                #         tf.summary.trace_export(
                #             name = 'critic_trace', step = 0, profiler_outdir = critic_log_dir
                #         )             
                
                if counter % self.n_critic == 0:
                    # if trace:
                    #     tf.summary.trace_on(graph = True, profiler = True)
                
                    g_loss = self.train_step_g(batch_size, hard)

                    # if trace:
                    #     with generator_summary_writer.as_default():
                    #         tf.summary.trace_export(
                    #             'generator_trace', step = 0, profiler_outdir = generator_log_dir
                    #             )
                    #     start = False    
                        
                counter += 1
            with critic_summary_writer.as_default():
                    tf.summary.scalar('loss', c_loss, step = epoch)
                    
            with generator_summary_writer.as_default():
                        tf.summary.scalar('loss', g_loss, step = epoch)
            if (epoch + 1) % 5 == 0:
                # Checkpooint functionality here      
                
                print('Time for epoch {} is {} sec \n with critic loss: {} and generator loss {}'.format(epoch + 1,
                                                                                                          time.time() - start, 
                                                                                                            c_loss, g_loss))
            if (temp_anneal):
                self.set_temperature(self.temperature-temp_increment)
            dataset = dataset.shuffle(buffer_size=10000)

    @tf.function
    def train_step_c(self, data_batch, hard):
        tot_dim = data_batch.shape[1]
        start_cat_dim = tot_dim - sum(self.cat_dims)
        noise = tf.random.normal((len(data_batch), self.latent_dim))

        with tf.GradientTape() as crit_tape:

            fake_data = self.generator(noise, training=True)
            if self.cat_dims != ():
                #print('test')
                fake_data = sample_gumbel(fake_data, self.temperature, self.cat_dims, hard)

            real_output = self.critic(data_batch, training=True)
            fake_output = self.critic(fake_data, training=True)

            crit_loss = critic_loss(real_output, fake_output)
            
            if self.mode == 'wgan-gp':
                gp_loss = self.gp_const * gradient_penalty(partial(self.critic), data_batch, fake_data)
                
                crit_loss += gp_loss

            critic_gradients = crit_tape.gradient(crit_loss , self.critic.trainable_variables)
            self.crit_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            self.c_loss(crit_loss)
        return crit_loss

    @tf.function
    def train_step_g(self, batch_size, hard):
        noise = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            gen_tape.watch(fake_data)

            if self.cat_dims != ():
                fake_data = sample_gumbel(fake_data, self.temperature, self.cat_dims, hard)

            fake_output = self.critic(fake_data, training=True)
            gen_loss = generator_loss(fake_output)
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

            self.gen_opt.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.g_loss(gen_loss)
        return gen_loss

    def set_temperature(self, temperature):
        self.temperature = temperature

    def save(self, path, force=False):
        """Save the fitted model at the given path."""
        if os.path.exists(path) and not force:
            print('The indicated path already exists. Use `force=True` to overwrite.')
            return

        base_path = os.path.dirname(path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print('Model saved successfully.')

