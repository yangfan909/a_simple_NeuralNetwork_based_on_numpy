# a_simple_NeuralNetwork_based_on_numpy

[TOC]

This is a simple neural network based on numpy constructed by myeslf.It maybe very rough.
__注意：本项目还未完成!__
## 目标

<font size = 20>仿照keras建造一个初步的神经网络架构。它将包括</font><br>
<table>
  <tr>
<td>激活函数</td><td>sigmoid，softmax，relu</td>
  </tr>
  <tr>
<td>网络层</td><td>全连接层，2D卷积层，2D最大池化层</td>
    </tr>
  <tr>
    <td>优化器</td><td>SGD随机梯度下降</td>
    </tr>
  <tr>
  <td>正则化</td><td>L2正则化</td>
  </tr>
  <tr>
    <td colspan = 2>模型导出与导入</td>
    </tr>
  </table>
  
  ## 可叠加模型
  
  本次构建尝试使将各个功能相分离，使得各个组件低耦合，高内聚，并且可以给予使用者一些自由，使其可以动手搭建属于自己的神经网络结构，而不限于代码提供的示例。
  
  实现目标如下:<br>
  <code>
  input = Tensor(inputshape)</code><br><code>
  convlotionOut = Convolution2D(filters,keneral_shape,activator)(input)</code><br><code>
  denseOut = Dense(neurons,activator)(convlotionOut)</code>

  
