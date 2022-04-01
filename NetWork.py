from Layer import*
from Activator import *
from abc import abstractmethod
import numpy as np
import random
from Layer import epsilon
def listExpend(x:np.array ,times:int = 0):
    x = x.tolist()
    y = x.copy()
    for i in range(1,times):
        for j in range(len(y)):
            x.append(y[j])
    for i in range(len(x)):
        x[i] = np.array(x[i])
    return x
class LOSSFunc(object):
    @abstractmethod
    def __init__(self,):
        pass
    @abstractmethod
    def __call__(self,f,res):
        pass
class Cross_Entropy(LOSSFunc):
    def __init__(self,):
        pass
    def __call__(self,f,res):
        f = np.array(f)
        res = np.array(res).flatten()
        # print("f\n",f,res)
        temp = np.log(f)
        los = -res*temp
        def reverse(x):
            # if 0< x and x<epsilon :
            #     x = epsilon
            # elif -epsilon<x and x<0:
            #     x = -epsilon
            y = x**-1 if x!=0 else 0.00001
            if y>10000:
                y = 10000
        
            return y
        self.res = np.frompyfunc(reverse,1,1)
        derivation = -res*self.res(f)
        derivation = np.array(derivation)
        derivation = derivation.reshape(1,derivation.shape[0])
        # print("insert1:\n",derivation)
        return derivation,sum(los.flatten())
        pass
class Model(object):
    def __init__(self,input:Tensor =None,output:Tensor =None):
        self.input = input
        self.output = output
        self.layer = output.getHiddenLayer()
    def compileLoss(self,lossFunc:LOSSFunc = None):
        self.lossFunc = lossFunc 
        pass
    def compileRegular(self,regularization:Regularization = None):
        self.regularization = regularization
        pass
    def compileOptimizer(self,):
        pass
    def predict(self,input):
        input = np.array(input)
        for item in self.layer:
            input = item.forward(input)
            # print("input:\n",input, "\n")
        return input
    def outPredict(self,input):
        output =[]
        for item in input:
            output.append(self.predict(item))
        return output
    def fit(self,xTrain,yTrain,epoch:int = None,BatchSize:int = None,log:bool = True):
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        input = 0
        output = 0
        feed = 0
        sumloss = 0
        loss = 0
        assert len(xTrain) * epoch% BatchSize == 0
        iteration =int( len(xTrain)*epoch/BatchSize )
        print("iteration",iteration)
        assert len(xTrain) * epoch == BatchSize * iteration
        xTrain = listExpend(xTrain,epoch)
        yTrain = listExpend(yTrain,epoch)
        index = list(range( len(xTrain)))
        random.shuffle(index)
        for i in range(iteration):
            for j in range(BatchSize):
                testIndex = index[i*BatchSize+j]
                input = xTrain[testIndex]
                input = self.predict(input)
                    #NOTICE: This is a temporary test
                # print("input:\n",input, "\n")
                derivation,loss = self.lossFunc(f = input,res = yTrain[testIndex])
                sumloss = sumloss + loss
                for item in self.layer[-1::-1]:
                    derivation = item.feedBackward(derivation)
                    derivation = np.array(derivation)
                    # print("derivation:\n",derivation) 
            for item in self.layer:
                item.parameterUpdate(BatchSize,self.regularization)
            print("iteration=%d,loss = %.8f"%(i,sumloss/BatchSize))
            if sumloss<0.2:
                return
            sumloss = 0
        pass