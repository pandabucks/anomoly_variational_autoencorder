from model import VAEModel
from keras.datasets import mnist
import numpy as np

input_shape = (32, 32, 3)

def main():
    print("loading model")
    model = VAEModel(input_shape)
    vae = model.load_model(input_shape)

    print("loading MNIST datasets")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # 数字1のデータを集める
    x_train_1 = []
    for i in range(len(x_train)):
        if y_train[i] == 1:
            x_train_1.append(x_train[i].reshape((28, 28, 1)))
    x_train_1 = np.array(x_train_1)
    x_train_1 = cut_img(x_train_1, 100000)
    print("train data:",len(x_train_1))

    # 評価データの作成
    x_test_1, x_test_9 = [], []
    for i in range(len(x_test)):
        if (y_test[i] == 1):  # Fashion mnist の場合は7（スニーカー）
            x_test_1.append(x_test[i].reshape((28, 28, 1)))
        if (y_test[i] == 9):
            x_test_9.append(x_test[i].reshape((28, 28, 1)))
    
    x_test_1 = np.array(x_test_1)
    x_test_9 = np.array(x_test_9)
    
    # test_normal画像(1)と　test_anomaly画像(9)をランダムに選ぶ  
    test_normal = x_test_1[np.random.randint(len(x_test_1))]
    test_anomaly = x_test_9[np.random.randint(len(x_test_9))]
    test_normal = test_normal.reshape(1, 28, 28, 1)
    test_anomaly = test_anomaly.reshape(1, 28, 28, 1)

def cut_img(self, x, number, height=8, width=8):
    print("cutting images ...")
    x_out = []
    x_shape = x.shape

    for i in range(number):
        shape_0 = np.random.randint(0,x_shape[0])
        shape_1 = np.random.randint(0,x_shape[1]-height)
        shape_2 = np.random.randint(0,x_shape[2]-width)
        temp = x[shape_0, shape_1:shape_1+height, shape_2:shape_2+width, 0]
        x_out.append(temp.reshape((height, width, x_shape[3])))

    print("Complete.")
    x_out = np.array(x_out)

    return x_out

if __name__ == "__main__":
    main()

