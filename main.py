from Data import *
from MLP import *
import time
from new_MLP import Layer, NEW_MLP
if __name__ == "__main__":
    d = Dataset()
    mlp = NEW_MLP(dataset=d, epoch=500, batch_size=500, lr=0.001, momentum=0.9)
    mlp.add(Layer(units=784, input_dim=200, activation='sigmoid'))
    mlp.add(Layer(200))
    mlp.add(Layer(100))
    mlp.add(Layer(50))
    mlp.add(Layer(units=10, activation='sigmoid'))
    mlp.compile()
    mlp.forward(d.train_x)
    for i in mlp.output_list:
        print(np.array(i).shape)
    # print(mlp.output_list)
    # mlp = MLP(dataset=d, epoch=500, batch_size=500, lr=0.0035, momentum=0.9)
    # start = time.time()
    # mlp.fit()
    # mlp.predict(d.train_x, d.train_Y)
    # print("train accuracy = {:.2f}%".format(mlp.accuracy))
    # mlp.predict(d.test_x, d.test_Y)
    # print("test accuracy = {:.2f}%".format(mlp.accuracy))
    # end = time.time()
    # print("cost time = {:.2f} sec".format(end - start))