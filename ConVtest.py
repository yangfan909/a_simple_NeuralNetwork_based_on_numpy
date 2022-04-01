from Layer import Tensor, Convolution2D, Flatten,Dense,MaxPooling2D
from NetWork import *
import numpy as np
from Activator import to_categorical,categorical_back
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
trainSize = len(x_train)
testSize = len(x_test)
# print('trainSize = %d, testSize = %d'%(trainSize, testSize))
# trainSize = 60000, testSize = 10000
x_train = np.array(x_train[0:1000])
x_test = np.array(x_test)
y_train_onehot = to_categorical(y_train[: 1000],10)
y_test_onehot = to_categorical(y_test,10)
print(y_test_onehot)
Input = Tensor(28,28)
# print(input)
cv2Dout = Convolution2D(filters=3,kernel=(3,3),activation='relu')(Input)
maxpoolingOut = MaxPooling2D(shape=(3,3))(cv2Dout)
flatOut = Flatten()(maxpoolingOut)
softmaxOut = Dense(neurons=10,activation="softmax")(flatOut)
model = Model(input = Input, output=softmaxOut)
model.compileLoss(Cross_Entropy())
model.compileRegular(L2Regularization(lamd=0.0001))
model.fit(x_train, y_train_onehot,BatchSize = 100,epoch = 2)
y_pre= model.outPredict(x_test)
y_decode = categorical_back(y_pre)
for i in range(50):
    print("pre:%d,test:%d" % (y_decode[i],y_test[i]))