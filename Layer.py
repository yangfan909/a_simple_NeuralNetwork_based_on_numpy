from re import S
import numpy as np
import Activator as Actor
from math import ceil
from abc import abstractmethod

class Layer(object):
    def __init__(self):
        pass
    @abstractmethod
    def get_layer(self):
        pass
    @abstractmethod
    def forward(self,input):
        pass
    @abstractmethod
    def feedBackward(self,output):
        pass
    def __call__(self,data):
        pass

class Tensor(object):
    def __init__(self,*args):
        self.shape = args
        self.layer = []
        pass
    def __call__(self,input = None):
        return self.data
    def addLayer(self,layer:Layer,temp= None):
        if temp != None:
            a = temp.getHiddenLayer()
            for i in a:
                self.layer.append(i)
        self.layer.append(layer)
    def getHiddenLayer(self)-> list[Layer]:
        return self.layer
    def get_layer(self):
        return self.shape
class Dense(Layer):#epsilon

    def __init__(self,inputShape:tuple= None, neurons:int= None, activation :str=None,learningRate :float=0.001,biasUsed :bool=False ,clipsize:float = 1,clipval:float = 1):
        self.learningRate = learningRate
        self.neurons = neurons
        self.activation = activation
        self.biasUsed = biasUsed
        self.bias = np.random.rand( self.neurons,1 )
        self.activation = Actor.interceptor(activation)
        self.clipval = clipval
        self.clipsize =clipsize
        def clipByValue(x):
            if x< -clipval:
                x = -clipval
            elif x>clipval:
                x = clipval
            return x
        self.clipval = np.frompyfunc(clipByValue,1,1)
        if(inputShape != None):
            self.lastDim = inputShape[-1]
            self.weights = np.random.rand(neurons,inputShape[-1])
            self.backPropagationW = np.zeros( (neurons,inputShape[-1]) )
            self.backPropagationB = np.zeros( (neurons,1) )
            self.backPropagation = np.zeros(inputShape[-1])
    def __call__(self, input:Tensor = None):

        self.lastDim = input.get_layer()[-1]
        self.inputShape = input.get_layer()

        self.weights = np.random.rand(self.neurons ,self.lastDim)
        self.backPropagationW = np.zeros( (self.neurons,self.inputShape[-1]) )
        self.backPropagationB = np.zeros( (self.neurons,1) )
        self.output = Tensor((self.neurons))
        self.output.addLayer(self,input)
        return self.output

    def forward(self,input):
        self.input =np.array(input)
        self.result =np.matmul(self.weights,self.input);

        # print(self.result,"\n",self.result.shape)
        if self.biasUsed:
            self.result += self.bias
        self.result = self.activation(self.result)
        '''
        div(ai,Wij) = div(ai,yi)*div(yi,Wij)
        y0  [W00,W01,W02,W03,W04,W05]  a0
        y1  [W10,W11,W12,W13,W14,W15]  a1
        y2 =[W20,W21,W22,W23,W24,W25] *a2
        y3  [W30,W31,W32,W33,W34,W35]  a3
                                       a4
                                       a5
        ...       ...
        '''              
        return self.result
    def feedBackward(self,feedback:np.array):
        self.loss_div_weights = np.zeros( (self.neurons,self.lastDim) )
        self.loss_div_bias = np.zeros( (self.neurons))
        self.loss_div_bias =np.matmul(feedback,self.activation.derivation(self.result))
        print('self.res_div_bias\n',self.loss_div_bias)

        print(self.input)
        # for i in range(self.neurons):
        #     for j in range(self.lastDim):
        #         self.loss_div_weights[i,j] += self.loss_div_bias[0,i]*\
        #         self.input[j]
        #         pass  
        self.input = np.array(self.input).reshape(1,self.lastDim)
        self.loss_div_weights =np.array(np.matmul(self.loss_div_bias.T,self.input))
        self.clipval(self.loss_div_weights)
        self.clipval(self.loss_div_bias)

        self.backPropagationW = self.backPropagationW + self.loss_div_weights
        if self.biasUsed:
            self.backPropagationB = self.backPropagationB + self.loss_div_bias
        self.backPropagation =np.matmul(feedback,self.weights)
        # print('self.res_div_bias\n',self.loss_div_bias)
        print('self.res_div_weights\n',self.loss_div_weights)

        return self.backPropagation

    def parameterUpdate(self,sum:int):
        self.weights -= self.learningRate/sum*self.backPropagationW
        self.bias -= self.learningRate/sum*self.backPropagationB
        self.backPropagationW -= self.backPropagationW
        self.backPropagationB -= self.backPropagationB
        pass
class Convolution2D(Layer):
    def __init__(self,neurons:int,shape:tuple, activation :str=None,learningRate: float=0.001,biasUsed :bool =False,clipsize:float = 0.1 ,clipval:float = 0.1):
        self.learningRate = learningRate
        self.neurons = neurons
        self.shape = shape
        self.activation = activation
        self.biasUsed = biasUsed
        self.activation = Actor.interceptor(activation)
        self.clipval = clipval
        self.clipsize =clipsize

    def __init__(self,inputShape:tuple = None, neurons:int= None,shape:tuple=None, activation :str=None,learningRate :float=0.001,biasUsed :bool=False ,clipsize:float = 0.001,clipval:float = 0):
        self.__init__(neurons,shape,activation,learningRate,biasUsed,clipsize,clipval)
        if len(inputShape)==2:
            inputShape =(1,inputShape[0],inputShape[1])
        self.inputShape = inputShape

        self.outShape = (self.neurons*self.inputShape[0],self.shape[0]-self.inputShape[1]+1,self.shape[1]-self.inputShape[2]+1)
        self.bias = np.random.rand( (neurons*inputShape[0],shape[0],shape[1]) )
        self.weights = np.random.rand( (neurons*inputShape[0],shape[0],shape[1]) )
        self.backPropagationW = np.zeros( (neurons*inputShape[0],shape[0],shape[1]) )
        self.backPropagationB = np.zeros( (neurons*inputShape[0],shape[0],shape[1]) )
    def __call__(self, input:Tensor = None):
        self.inputShape = input.get_layer()
        self.outShape = (self.neurons*self.inputShape[0],self.inputShape[1]-self.shape[0]+1,self.inputShape[2]-self.shape[1]+1)
        self.output = Tensor(self.outShape)
        self.output.addLayer(self,temp=input)
        return self.output

    def forward(self,input):
        self.input =np.array(input)

        self.result = np.zeros(self.outShape)
        for i0 in range(self.inputShape[0]):
            for i1 in range(self.neurons):
                for j in range(self.output.shape[1]):
                    for k in range(self.output.shape[2]):
                        for l in range(self.shape[0]):
                            for m in range(self.shape[1]):
                                self.result[i0*self.neurons+i1,j,k] += self.weights[i1,l,m]*self.input[i1,j+l,k+m] 

        if self.biasUsed:
            self.result += self.bias
        self.result = self.activation(self.result)
        self.act_div_res = self.activation.derivation(self.result)
        self.res_div_weights = np.zeros_like(self.weights)
        return self.result
    def feedBackward(self,feedback:np.array):
        self.res_div_weights = np.matmul(feedback,self.act_div_res)
        for i in range(self.outShape.shape[0]):
            for j in range(self.outShape.shape[1]):
                for k in range(self.outShape.shape[2]):
                    for l in range(self.shape[0]):
                        for m in range(self.shape[1]):
                            self.res_div_weights[i][l][m] += self.res_div_weights[i,j,k]*self.input[int(i/self.neurons),j+l,k+m]
                            pass

        for i in range(self.outShape[0]):
            for j in range(self.shape[0]):
                for k in range(self.shape[1]):
                    if self.res_div_weights[i][j][k] > self.clipval:
                        self.res_div_weights[i][j][k] = self.clipval
                    elif self.res_div_weights[i][j][k] <-self.clipval:
                        self.res_div_weights[i][j][k] = -self.clipval
        self.backPropagationW += self.res_div_bias
        # if self.biasUsed:
        #     self.backPropagationB += tempB

    def parameterUpdate(self,sum:int):
        self.weights -= self.learningRate/sum*self.backPropagationW
        self.bias -= self.learningRate/sum*self.backPropagationB
        self.backPropagationW -=self.backPropagationW
        self.backPropagationB -= self.backPropagationB
        pass
class MaxPooling2D(Layer):
    def __init__(self,shape:tuple,input:Tensor = None):
        self.inputShape = input
        self.shape = shape
    def __call__(self, input:Tensor = None):
        self.inputShape = input.get_layer()
        self.outputShape = (self.inputShape[0],ceil(self.inputShape[1]/self.shape[0])\
            ,ceil(self.inputShape[2]/self.shape[1]))
        self.output = Tensor(self.outputShape)
        self.output.addLayer(self)
        return self.output

    def forward(self,input):
        input = np.array(input)
        print(input.shape)
        self.res = np.zeros(self.outputShape)
        self.feedback = np.zeros(input.shape)
        for i in range(self.outputShape[0]):
            for j in range(self.outputShape[1]):
                for k in range(self.outputShape[2]):
                    print(self.shape,self.inputShape)
                    temp = input[ i,j*self.shape[0]:min((j+1)*self.shape[0],self.inputShape[1]),k*self.shape[1]:min((k+1)*self.shape[1],self.inputShape[2])]
                    print(temp)
                    print("temp.shape",temp.shape)
                    index = np.unravel_index(temp.argmax(),temp.shape)
                    print(index)
                    self.res[i][j][k] = max(temp.flatten() )
                    self.feedback[i][j*self.shape[0]+index[0]][k*self.shape[1]+index[1]] = 1
                    pass
        return self.res,self.feedback
    def feedBackward(self,feedback:np.array):
       return feedback*self.feedback

    def parameterUpdate(self,sum:int):
     
        pass