from abc import abstractmethod
import numpy as np
class Activator(object):
    def __init__(self,):
        pass
    @abstractmethod
    def __call__(self, x):
        pass
    @abstractmethod
    def derivation(self):
        pass
class Logistic(Activator):
    def __init__(self):
        pass
    def __call__(self, x):
        self.x = x
        self.y = (np.exp(-self.x)+1)**-1
        return self.y
    def derivation(self,res,all = True):
        if all:
            self.res = (res)*(1-res)
            self.res = np.diag(self.res)
        else:
            self.res = (res)*(1-res)
        return self.res
class ReLu(Activator):
    def __init__(self,gamma=0.01):
        self.gamma = gamma
        def func(x):
            return x if x>0 else self.gamma*x
        def div(x):
            return 1 if x>0 else self.gamma
        self.func = func
        self.func = np.frompyfunc(func,1,1)
        self.div = np.frompyfunc(div,1,1)
        pass
    def __call__(self,x):
        return self.func(x)
    def derivation(self,y,all = True):
        if all:
            self.res = np.diag(self.div(y))
        else:
            self.res = self.div(y)
        return self.res
class Softmax(Activator):
    def __init__(self):
        pass
    def __call__(self,x):
        self.x = np.array(x).flatten()
        self.temp = []
        for i in self.x:
            self.temp.append(np.exp(i))
        self.sum = sum(self.temp)    
        self.y = []
        for i in self.temp:
            self.y.append( i/self.sum)
        self.y = np.array(self.y)
        return self.y
    def derivation(self,res):
        self.res = np.zeros((len(self.x), len(self.x)))
        size = len(self.x)
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.res[i, i] = np.exp(self.x[i])*(self.sum-self.x[i])/self.sum ** 2
                else:
                    self.res[i, j] = -np.exp(self.x[j])*np.exp(self.x[i]) / self.sum ** 2
        return self.res
def interceptor(input:str = "logistic")->Activator:
    activator = Activator()
    if input == "logistic":
        activator = Logistic()
    elif input == "relu":
        activator = ReLu()
    elif input == "softmax":
        activator = Softmax()
    return activator
# x = [1,2,3,4,5]
# activator = Softmax()
# res = activator(x)
# print("res:", res)
# print("activator:",activator.derivation())