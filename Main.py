import sys

print("参数个数:", len(sys.argv))
print("参数列表:", sys.argv)
print("程序名:", sys.argv[0])

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
print(y_test_onehot[0:10])
Input = Tensor(28, 28)
out = Input
list1 = sys.argv
outpath = list1[1]
with open(outpath, 'w+') as file:
    for item in list1:
        file.write(item + " ")
    file.write("\n")
for i in range(2, len(list1)):
    if list1[i] == 'Dense':
        n = int(list1[i + 1])
        a = list1[i + 2]
        out = Dense(neurons=n, activation=a)(out)
    elif list1[i] == 'Maxpool':
        l = int(list1[i + 1])
        m = int(list1[i + 2])
        out = MaxPooling2D(shape=(l, m))(out)
    elif list1[i] == 'Conv':
        f = int(list1[i + 1])
        l = int(list1[i + 2])
        m = int(list1[i + 3])
        a = list1[i + 4]
        out = Convolution2D(filters=f, kernel=(l, m), activation=a)(out)
    elif list1[i] == 'Flat':
        out = Flatten()(out)
model = Model(input=Input, output=out)
model.compileLoss(Cross_Entropy())
model.compileRegular(L2Regularization(lamd=0.01))
optimizer = SGD(lr=0.001, decay=0.999, clipvalue=10)
model.compileOptimizer(optimizer)

now = datetime.now()
print("begin", now.strftime("%y/%m/%d/%H:%M"))

model.fit(x_train, y_train_onehot, iteration=3000, filename=outpath, xtest=x_test[0:1000], ytest=y_test_onehot[0:1000], step=100)
model.logModel()
y_pre = model.outPredict(x_test)
y_decode = categorical_back(y_pre)

with open(outpath, 'a') as file:
    for i in range(0, 500):
        file.write("pre:%d,test:%d\n" % (y_decode[i], y_test[i]))

now = datetime.now()
print("end", now.strftime("%y/%m/%d/%H:%M"))
