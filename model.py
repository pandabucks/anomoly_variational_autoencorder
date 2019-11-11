import keras
from keras.layers import Lambda, Input, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.datasets import fashion_mnist

from keras import backend as K
from keras.layers import BatchNormalization, Activation, Flatten
from keras.layers.convolutional import Conv2DTranspose, Conv2D

latent_dim = 2
Nc = 16

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
class VAEModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def load_model(self, input_shape):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Conv2D(Nc, kernel_size=2, strides=2)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*Nc, kernel_size=2, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        #encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(2*2)(latent_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((2,2,1))(x)
        x = Conv2DTranspose(2*Nc, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(Nc, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x1 = Conv2DTranspose(1, kernel_size=4, padding='same')(x)
        x1 = BatchNormalization()(x1)
        out1 = Activation('sigmoid')(x1)#out.shape=(n,8,8,1)

        x2 = Conv2DTranspose(1, kernel_size=4, padding='same')(x)
        x2 = BatchNormalization()(x2)
        out2 = Activation('sigmoid')(x2)#out.shape=(n,8,8,1)

        decoder = Model(latent_inputs, [out1, out2], name='decoder')
        #decoder.summary()

        # build VAE model
        outputs_mu, outputs_sigma_2 = decoder(encoder(inputs)[2])
        vae = Model(inputs, [outputs_mu, outputs_sigma_2], name='vae_mlp')

        # VAE loss
        m_vae_loss = (K.flatten(inputs) - K.flatten(outputs_mu))**2 / K.flatten(outputs_sigma_2)
        m_vae_loss = 0.5 * K.sum(m_vae_loss)

        a_vae_loss = K.log(2 * 3.14 * K.flatten(outputs_sigma_2))
        a_vae_loss = 0.5 * K.sum(a_vae_loss)

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(kl_loss + m_vae_loss + a_vae_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        return vae


