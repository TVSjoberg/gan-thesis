import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class ClipConstraint(keras.constraints.Constraint):
    # Enforces clipping constraints in WGAN

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return keras.backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


def sample_gumbel(logits, temperature, cat_dims=(), hard=False):
    start_dim = tf.shape(logits)[1] - sum(cat_dims)
    for dim in cat_dims:  # Draw gumbel soft-max for each categorical variable
        temp_logits = logits[:, start_dim:start_dim + dim]
        dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=temp_logits)
        temp_logits = dist.sample()

        if hard:  # make One_hot
            temp_logits = tf.one_hot(tf.math.argmax(temp_logits, axis=1), dim)

        logits = tf.concat([logits[:, :start_dim], temp_logits, logits[:, start_dim + dim:]], axis=1)
        # logits[:, start_dim:start_dim+dim] = temp_logits

        start_dim += dim
    return logits


def gradient_penalty(f, real_data, fake_data):
    alpha = tf.random.uniform(shape=[real_data.shape[0], 1])

    inter = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
    gp = tf.reduce_mean((slopes - 1) ** 2)

    return gp


def wasserstein_loss(y_real, y_critic):
    return keras.backend.mean(y_real * y_critic)


def critic_loss(real_output, fake_output):
    real_loss = wasserstein_loss(real_output, -tf.ones_like(real_output))
    fake_loss = wasserstein_loss(fake_output, tf.ones_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return wasserstein_loss(-tf.ones_like(fake_output), fake_output)
