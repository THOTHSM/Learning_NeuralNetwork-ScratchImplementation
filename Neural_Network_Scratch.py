import numpy as np

class dense_layer:
    def __init__(self,input_size,number_of_neurons):
        
        """input_size : number of features in inpput to the neoran, 
            number_of_neurons : Neurons in each layer 
        """
        self.weights = 0.01*np.random.randn(input_size,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))
    
    def forward(self,input_data):
        self.input_data = input_data
        self.output = np.dot(input_data,self.weights)+self.bias
    
    def backward(self,dvalues):
        self.dweights = np.dot(self.input_data.T,dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class relu_activation:
    """Has forward and backward function"""
    def forward(self,input_data):
        """input_data : data as input"""
        self.input_data = input_data
        self.output = np.maximum(0,input_data)
    
    def backward(self,dvalues):
        """dvalues : derivative matrix with all the front layers with respect to loss """
        self.dinputs = dvalues.copy()
        self.dinputs[self.input_data<=0]=0

class softmax_activation:
    def forward(self,input):
        exp_input = np.exp(input - np.max(input,axis=1,keepdims=True))
        self.output = exp_input / np.sum(exp_input,axis=1,keepdims=True)

class Loss:
    def calculate(self,y_pred,y_actual):
        negative_log_likelihood = self.forward(y_pred,y_actual)
        loss = np.mean(negative_log_likelihood)
        return loss


class Categorical_cross_entropy(Loss):
    def forward(self,y_pred,y_actual):
        self.y_pred_clip = np.clip(y_pred,1e-7,1-1e-7)

        if y_actual.ndim==1:
            correct_confidence = self.y_pred_clip[range(len(y_pred)),y_actual]
        elif y_actual.ndim==2:
            correct_confidence = np.sum(y_actual*self.y_pred_clip,axis=1)
        else:
            raise ValueError("y_actual should be 1-dimensional (label encoded) or 2-dimensional (one-hot encoded).")
        
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

class softmax_categoricalcrossentropy:
    def __init__(self):
        self.softmax = softmax_activation()
        self.cross_entropy = Categorical_cross_entropy()

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

class metric:
    def accuracy(self,final_output,y_actual):
        final_prediction = np.argmax(final_output,axis=1)

        if y_actual.ndim==2: # y is onehot encodeed
            actual_prediction = np.argmax(y_actual,axis=1)
        elif y_actual.ndim==1:
            actual_prediction = y_actual
        else:
            raise ValueError("y_actual should be 1-dimensional (label encoded) or 2-dimensional (one-hot encoded).")
         
        accu = np.mean(final_prediction==actual_prediction)
        return accu

class SGD_Optimizer:
    def __init__(self,learning_rate) -> None:
        self.learning_rate = learning_rate

    def update_para(self,layer):
        layer.weights +=  (-self.learning_rate * layer.dweights)
        layer.bias += (-self.learning_rate * layer.bias)


