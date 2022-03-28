from Layer import*
from Activator import *
from abc import abstractmethod
import numpy as np
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
        res = np.array(res)
        f = np.log(f)
        los = -res*f
        return sum(los.flatten())
        pass
class model(object):
    def __init__(self,input:Tensor =None,output:Tensor =None):
        self.input = input
        self.output = output
        self.layer = out.getHiddenLayer()
    def compileLoss(self,):
        pass
    def compileRegular(self,):
        pass
    def compileOptimizer(self,):
        pass
    def fit(self,xTrain,yTrain,epoch:int = None,iteration:int = None,BatchSize:int = None,log:bool = True):
        pass