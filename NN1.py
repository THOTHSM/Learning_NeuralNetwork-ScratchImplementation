import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()


class dense_layer:
    def __init__(self, input_feature_size, number_of_neurons):
        self.input_feature_size = input_feature_size
        self.number_of_neurons = number_of_neurons
        self.weights = 0.01*np.random.randn(input_feature_size,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))
    
    def forward_value(self,input_data):
        self.output = np.dot(input_data,self.weights)+self.bias
        

x,y = spiral_data(samples=100,classes=3)

dense1=dense_layer(2,3)

dense1.forward_value(x)

print(dense1.output)