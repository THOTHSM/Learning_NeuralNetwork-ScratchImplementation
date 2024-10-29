import numpy as np

class dense_layer:
    def __init__(self, input_feature_size, number_of_neurons):
        self.input_feature_size = input_feature_size
        self.number_of_neurons = number_of_neurons
        self.weights = 0.01*np.random.randn(input_feature_size,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))
    
    def forward(self,input_data):
        self.output = np.dot(input_data,self.weights)+self.bias

    def backward(self,dvalues,input_data):
        self.inputs = input_data
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)


class relu_activation:
    def forward(self,input):
        self.input = input
        self.output = np.maximum(0,self.input)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input<=0]=0

class softmax_activation:
    def forward(self,input):
        exp_input = np.exp(input - np.max(input,axis=1,keepdims=True))
        self.output = exp_input / np.sum(exp_input,axis=1,keepdims=True)

class loss:
    def calculate(self,y_pred,y_actual):

        losses = self.forward(y_pred,y_actual) 
        cost = np.mean(losses)
        return cost 
    
class categorical_cross_entropy(loss):
    def forward(self,y_pred,y_actual):
        
        y_pred_clip = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_actual.shape) ==1: # This meaans y is in the form of target encoded

            correct_confidences = y_pred_clip[range(len(y_pred)),y_actual]
        
        elif len(y_actual.shape)==2: # This means y is one hot encoded

            correct_confidences = np.sum(y_pred*y_actual,axis=1)
        else:
            raise ValueError("y_actual should be 1-dimensional (label encoded) or 2-dimensional (one-hot encoded).")
        
        negative_log_likelihood = -np.log(correct_confidences)

        return negative_log_likelihood
    
    def backward_(self,y_pred,y_actual):
        number_of_data = len(y_pred)
    
        if y_actual.ndim == 1:
            number_of_class = y_pred.shape[1]
            y_actual = np.eye(number_of_class)[y_actual] # Converting to one hot encoding
        
        self.dinputs = -y_actual/y_pred
        self.dinputs = self.dinputs/number_of_data # Normalizing all the derivatives


class softmax_categoricalcrossentropy:
    def __init__(self):
        self.softmax = softmax_activation()
        self.cross_entropy = categorical_cross_entropy()

    def forward(self,inputs,y_actual):
        self.softmax.forward(inputs)
        self.outputs = self.softmax.output
        return self.cross_entropy.calculate(self.outputs,y_actual)
        
    def backward(self,y_pred,y_actual):
        number_of_data = len(y_pred)
    
        if y_actual.ndim == 1:
            number_of_class = y_pred.shape[1]
            y_actual = np.eye(number_of_class)[y_actual] # Converting to one hot encoding
        
        self.dinputs = y_pred-y_actual
        self.dinputs = self.dinputs/number_of_data

