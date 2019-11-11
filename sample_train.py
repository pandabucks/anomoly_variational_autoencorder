from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
from keras.layers import Lambda, Input, Dense, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from keras.layers import BatchNormalization, Activation, Flatten
from keras.layers.convolutional import Conv2DTranspose, Conv2D
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
 
 
# 8×8のサイズに切り出す関数
def cut_img(x, number, height=8, width=8):
    print("cutting images ...")
    x_out = []
 
    for i in range(number):
        shape_0 = np.random.randint(0,x.shape[0])
        shape_1 = np.random.randint(0,x.shape[1]-height)
        shape_2 = np.random.randint(0,x.shape[2]-width)
        temp = x[shape_0, shape_1:shape_1+height, shape_2:shape_2+width, 0]
        x_out.append(temp.reshape((8, 8, 1)))
    print("Complete.")
    x_out = np.array(x_out)
 
    return x_out
 
# データセットの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
 
 
# 学習データの作成
x_train_1 = []
for i in range(len(x_train)):
  if y_train[i] == 1:
     x_train_1.append(x_train[i].reshape((28, 28, 1)))
 
x_train_1 = np.array(x_train_1)
  # 8×8のサイズに切り出す関数を呼び出す
x_train_1 = cut_img(x_train_1, 100000)
print("train data:",len(x_train_1))
 
 
# 評価データの作成
x_test_1, x_test_9 = [], []
for i in range(len(x_test)):
  if y_test[i] == 1:
     x_test_1.append(x_test[i].reshape((28, 28, 1)))
  if y_test[i] == 9:
     x_test_9.append(x_test[i].reshape((28, 28, 1)))
 
x_test_1 = np.array(x_test_1)
x_test_9 = np.array(x_test_9)
 
  # test_normal画像(1)と　test_anomaly画像(9)をランダムに選ぶ  
test_normal = x_test_1[np.random.randint(len(x_test_1))]
test_anomaly = x_test_9[np.random.randint(len(x_test_9))]
test_normal = test_normal.reshape(1, 28, 28, 1)
test_anomaly = test_anomaly.reshape(1, 28, 28, 1)
 
 
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
 
# network parameters
input_shape=(8, 8, 1)
batch_size = 128
latent_dim = 2
epochs = 10  ###
Nc = 16
 
# build encoder model
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
encoder.summary()
 
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
decoder.summary()
 
# build VAE model
outputs_mu, outputs_sigma_2 = decoder(encoder(inputs)[2])
vae = Model(inputs, [outputs_mu, outputs_sigma_2], name='vae_mlp')
vae.summary()
 
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
 
# train the autoencoder
vae.fit(x_train_1,
        epochs=epochs,
        batch_size=batch_size)
        #validation_data=(x_test, None))
vae.save_weights('vae_mlp_mnist.h5')
 
#ヒートマップの計算
def evaluate_img(model, x_normal, x_anomaly, name, height=8, width=8, move=2):  #######
    img_normal = np.zeros((x_normal.shape))
    img_anomaly = np.zeros((x_normal.shape))
 
    for i in range(int((x_normal.shape[1]-height)/move)):
        for j in range(int((x_normal.shape[2]-width)/move)):
            x_sub_normal = x_normal[0, i*move:i*move+height, j*move:j*move+width, 0]
            x_sub_anomaly = x_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0]
            x_sub_normal = x_sub_normal.reshape(1, height, width, 1)
            x_sub_anomaly = x_sub_anomaly.reshape(1, height, width, 1)
 
            #従来手法
            if name == "old_":
                #正常のスコア
                normal_score = model.evaluate(x_sub_normal, batch_size=1, verbose=0)
                img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  normal_score
 
                #異常のスコア
                anomaly_score = model.evaluate(x_sub_anomaly, batch_size=1, verbose=0)
                img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  anomaly_score
 
            #提案手法
            else:
                #正常のスコア
                mu, sigma = model.predict(x_sub_normal, batch_size=1, verbose=0)
                loss = 0
                for k in range(height):
                    for l in range(width):
                        loss += 0.5 * (x_sub_normal[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
                img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss
 
                #異常のスコア
                mu, sigma = model.predict(x_sub_anomaly, batch_size=1, verbose=0)
                loss = 0
                for k in range(height):
                    for l in range(width):
                        loss += 0.5 * (x_sub_anomaly[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
                img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss
 
    save_img(x_normal, x_anomaly, img_normal, img_anomaly, name)
 
#ヒートマップの描画
def save_img(x_normal, x_anomaly, img_normal, img_anomaly, name):
    path = 'images/'
    if not os.path.exists(path):
          os.mkdir(path)
 
    #　※注意　評価したヒートマップを1～10に正規化
    img_max = np.max([img_normal, img_anomaly])
    img_min = np.min([img_normal, img_anomaly])
    img_normal = (img_normal-img_min)/(img_max-img_min) * 9 + 1
    img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1
 
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x_normal[0,:,:,0], cmap='gray')
    plt.axis('off')
    plt.colorbar()
 
    plt.subplot(2, 2, 2)
    plt.imshow(img_normal[0,:,:,0], cmap='Blues',norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()
    plt.clim(1, 10)
 
    plt.title(name + "normal")
 
    plt.subplot(2, 2, 3)
    plt.imshow(x_anomaly[0,:,:,0], cmap='gray')
    plt.axis('off')
    plt.colorbar()
 
    plt.subplot(2, 2, 4)
    plt.imshow(img_anomaly[0,:,:,0], cmap='Blues',norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()
    plt.clim(1, 10)
 
    plt.title(name + "anomaly")
 
    plt.savefig(path + name +".png")
    plt.show()
    plt.close()
 
 
#従来手法の可視化
evaluate_img(vae, test_normal, test_anomaly, "old_")
 
#提案手法の可視化
evaluate_img(vae, test_normal, test_anomaly, "new_")
 