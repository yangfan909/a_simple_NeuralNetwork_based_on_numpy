from Layer import MaxPooling2D, Tensor, Convolution2D, Dense
from Activator import *
import numpy as np
from math import ceil
tensor = Tensor(20)
tensor = Dense(neurons=5, activation="logistic")(tensor)
# maxpooling = MaxPooling2D(shape=(2,2))(tensor)
a = [[[2, 4, 7, 0, -11],
     [2, 5, 9, -34, 0],
     [-1, 3, 4, -90, 1],
     [-1, 3, 9, 34, 1]]]
a = np.array(a)
layer = tensor.getHiddenLayer()
aFlat = np.array(a)
aFlat = aFlat.flatten()
aFlat = list(aFlat)
res = layer[0].forward(aFlat)
print("res", res)
fed = layer[0].feedBackward(np.ones((1, 5)))
print("fed", fed)
index = np.zeros_like(a)

CvShape = (2, 2)
InputShape = a.shape
outshape = (ceil(InputShape[0]/CvShape[0]), ceil(InputShape[1]/CvShape[1]))
b = np.zeros(outshape)
for i in range(outshape[0]):
    for j in range(outshape[1]):
        temp = a[i*CvShape[0]: min((i+1) * CvShape[0], InputShape[0]),
                 j*CvShape[1]: min((j+1) * CvShape[1], InputShape[1])]
        index0 = np.unravel_index(temp.argmax(), temp.shape)
        print("argmax", index0)
        print(i, j, "\n", temp)
        b[i][j] = max(temp.flatten())
print("b:", b)
