from Layer import Tensor, Convolution2D, Flatten, Dense, MaxPooling2D
from NetWork import *
import numpy as np
from Activator import to_categorical, categorical_back
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
trainSize = len(x_train)
testSize = len(x_test)
# print('trainSize = %d, testSize = %d'%(trainSize, testSize))
# trainSize = 60000, testSize = 10000
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)
print(y_test_onehot)

Input = Tensor(28, 28)
out = Convolution2D(filters=4, kernel=(4, 4), activation='relu', biasUsed=True)(Input)
out = MaxPooling2D(shape=(2, 2))(out)
out = Convolution2D(filters=8, kernel=(3, 3), activation='relu')(out)
out = Flatten()(out)
out = Dense(neurons=50, activation='relu')(out)
softmaxOut = Dense(neurons=10, activation="softmax")(out)
optimizer = SGD(lr=0.001, decay=1.0, clipvalue=10)

model = Model(input=Input, output=softmaxOut)
model.compileLoss(Cross_Entropy())
model.compileRegular(L2Regularization(lamd=0.001))
model.compileOptimizer(optimizer)
model.fit(x_train, y_train_onehot, iteration=4000, filename='./test/2/', xtest=x_test[0:1000], ytest=y_test_onehot[0:1000], step=100)

path = model.logModel('./test/2/model.h5')
model.load_model(path)
print(model.evaluate(x_test, y_test_onehot))
y_pre = model.outPredict(x_test)
y_decode = categorical_back(y_pre)
print('ok')