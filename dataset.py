import numpy as np

from keras.datasets import mnist
from keras.datasets import fashion_mnist

class VAEDataset:
    def __init__(self):
        print("vae model")

    def load_train(self):
        print("load train datas")

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