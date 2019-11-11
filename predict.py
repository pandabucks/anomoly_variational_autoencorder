import keras
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

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

#ヒートマップの計算
def evaluate_img(model, x_normal, x_anomaly, name, height=8, width=8, move=2):
    img_normal = np.zeros((x_normal.shape))
    img_anomaly = np.zeros((x_normal.shape))

    for i in range(int((x_normal.shape[1]-height)/move)+1):
        for j in range(int((x_normal.shape[2]-width)/move)+1):
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



vae = keras.models.load_model('vae_mlp_mnist.h5', compile=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_test_1 = []
x_test_9 = []

for i in range(len(x_test)):
    if y_test[i] == 1:
        temp = x_test[i,:,:,:]
        x_test_1.append(temp.reshape((x_train.shape[1], x_train.shape[2], x_train.shape[3])))

    if y_test[i] == 9:
        temp = x_test[i,:,:,:]
        x_test_9.append(temp.reshape((x_train.shape[1], x_train.shape[2], x_train.shape[3])))

x_test_1 = np.array(x_test_1)
x_test_9 = np.array(x_test_9)

#正常/異常のテストデータ
idx1 = np.random.randint(len(x_test_1))
idx2 = np.random.randint(len(x_test_9))

test_normal = x_test_1[idx1,:,:,:]
test_anomaly = x_test_9[idx2,:,:,:]

test_normal = test_normal.reshape(1, test_normal.shape[0], test_normal.shape[1], test_normal.shape[2])
test_anomaly = test_anomaly.reshape(test_normal.shape)

#従来手法の可視化
evaluate_img(vae, test_normal, test_anomaly, "old_")

#提案手法の可視化
evaluate_img(vae, test_normal, test_anomaly, "new_")
