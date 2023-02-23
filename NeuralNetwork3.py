import sys
import numpy as np

class MLP:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.output = np.zeros(2)
        
        self.widths = []
        self.activation_funcs = []
        self.weights = []
        self.bias = []
        self.layers = []
        
        self.momentum_weights = []
        self.momentum_bias = []
        
    def Dense(self, n_neurons, activation_func):
        #input size for this layer, output_len = n_neurons
        prev_n_neurons = self.widths[-1] if self.widths else self.X.shape[1]
        # -- adjustment:() or not? normal or rand? -- 
        weights = np.random.rand(n_neurons, prev_n_neurons) * np.sqrt(2 / (n_neurons + prev_n_neurons))
        #-- bias Initialize to 0s -- 
        bias = np.zeros((n_neurons,1))
        layer = np.random.rand(n_neurons)
        self.widths.append(n_neurons)
        self.activation_funcs.append(activation_func)
        self.weights.append(weights)
        self.bias.append(bias)
        self.layers.append(layer)
        
        return self
    
    def OutputLayer(self):
        prev_n_neurons = self.widths[-1]
        weights = np.random.rand(self.output.shape[0], prev_n_neurons) * np.sqrt(2 / (self.output.shape[0] + prev_n_neurons))
        bias = np.zeros((2,1))
        
        self.weights.append(weights)
        self.bias.append(bias)
        
        return self
        
    def Momentum(self):
        for i in range(len(self.widths)):
            weights = np.zeros(self.weights[i].shape)
            bias = np.zeros(self.bias[i].shape)
            
            self.momentum_weights.append(weights)
            self.momentum_bias.append(bias)
            
        output_momentum_weights = np.zeros(self.weights[-1].shape)
        output_momentum_bias = np.zeros(self.bias[-1].shape)
        self.momentum_weights.append(output_momentum_weights)
        self.momentum_bias.append(output_momentum_bias)
        
        return self
    
    def Forward(self, X):
        for i in range(len(self.widths)):
            width = self.widths[i]
            weights = np.asarray(self.weights[i])
            bias = np.asarray(self.bias[i])
            activation_func = self.activation_funcs[i]
            if activation_func == 'relu':
                output = self.ReLU(np.matmul(weights, X.T) + bias)
            elif activation_func == 'sigmoid':
                output = self.Sigmoid(np.matmul(weights, X.T) + bias)
            output = output.T
            self.layers[i] = output
            X = output
        output_weights = np.asarray(self.weights[-1])
        output_bias = np.asarray(self.bias[-1])
        self.output = self.Softmax(np.matmul(output_weights, X.T) + output_bias)
        
        return self, self.output
    
    def Back(self, output, real_labels, learning_rate, beta, X):
        
        error = np.asarray(output - real_labels)
        
        output_delta_weights = np.matmul(error.T, self.layers[-1]) * (1./output.shape[0])
        output_delta_bias = np.sum(error, axis=0, keepdims=True).T * (1./output.shape[0])
        
        self.momentum_weights[-1] = beta * np.asarray(self.momentum_weights[-1]) + (1. - beta) * output_delta_weights
        self.momentum_bias[-1] = beta * np.asarray(self.momentum_bias[-1])  + (1. - beta) * output_delta_bias
        
        self.weights[-1] = np.asarray(self.weights[-1]) - learning_rate * np.asarray(self.momentum_weights[-1])
        self.bias[-1] = np.asarray(self.bias[-1]) - learning_rate * np.asarray(self.momentum_bias[-1])
        
        error = error.T
        delta_layer = error
        for i in range(len(self.widths)-1, -1, -1):
            layer = np.asarray(self.layers[i].T)
            delta_layer = np.matmul(self.weights[i+1].T, delta_layer)* layer * (1 - layer)
            
            if i == 0: 
                delta_weights = np.matmul(delta_layer, X) * (1./output.shape[0])
            else:
                delta_weights = np.matmul(delta_layer, self.layers[i-1]) * (1./output.shape[0])
                
            delta_bias = np.sum(delta_layer, axis=1, keepdims=True) * (1./output.shape[0])
            
            self.weights[i] = np.asarray(self.weights[i]) - learning_rate * self.momentum_weights[i]
            self.bias[i] = np.asarray(self.bias[i]) - learning_rate * self.momentum_bias[i]
            
            self.momentum_weights[i] = beta * np.asarray(self.momentum_weights[i]) + (1. - beta) * delta_weights
            self.momentum_bias[i] = beta * np.asarray(self.momentum_bias[i])  + (1. - beta) * delta_bias
        
        return self 
    '''
    def Predict(self, X, y):
        preds = []
        acc_cnt = 0
        for i in range(X.shape[0]):
            _, pred = self.Forward(X[i].reshape(1, X.shape[1]))
            preds.append(np.where(pred == np.amax(pred), 1., 0.).T[0])
        for i in range(len(preds)):
            if np.array_equal(preds[i], y[i]):
                acc_cnt += 1
        acc = acc_cnt / y.shape[0]
        return preds, acc
    '''
    def OutputPreds(self, X):
        preds = []
        for i in range(X.shape[0]):
            _, pred = self.Forward(X[i].reshape(1, X.shape[1]))
            preds.append(np.where(pred == np.amax(pred), 1., 0.).T[0])
        preds = [int(x[1]) for x in preds]
        return preds
    
    def Training(self, epochs, batch_size, learning_rate, beta):
        batches = self.X.shape[0] // batch_size
        
        for epoch in range(epochs):
            X_train, y_train = self.shuffle(self.X, self.y)
            avg_cost = 0
            cnt = 0
            for batch in range(batches):
                cnt += 1
                start = batch * batch_size
                end = min(start + batch_size, X_train.shape[0]-1)
                if start < end:
                    X_batch, y_batch = X_train[start:end], y_train[start:end]
                    _, output = self.Forward(np.asarray(X_batch))
                    output = output.T
                    train_cost = self.CrossEntropyLoss(output, y_batch)
                    avg_cost += train_cost
                    self.Back(output, y_batch, learning_rate, beta, np.asarray(X_batch))
            
        return self
    
    #activation functions: ReLU, Sigmoid
    def ReLU(self, z):
        return z * (z > 0)

    def Sigmoid(self, z):
        return 1. / (1 + np.exp(-z))

    #for output layer
    def Softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis = 0) 
    
    #shuffle function
    def shuffle(self, a, b):   
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def CrossEntropyLoss(self, preds, true_labels):
        return  - (1 / preds.shape[0]) * np.sum(np.multiply(true_labels, np.log(preds)))
    
if __name__ == "__main__":
    X_train = np.loadtxt(sys.argv[1], delimiter = ',')
    Y_train = np.loadtxt(sys.argv[2], dtype = 'int')
    X_test = np.loadtxt(sys.argv[3], delimiter = ',')
    
    Y_train = np.eye(2)[Y_train]
    
    model = MLP(X_train, Y_train)
    model.Dense(64, 'sigmoid')
    model.OutputLayer()
    model.Momentum()
    model = model.Training(200, 1, .04, 0.9)
    
    preds = model.OutputPreds(X_test)
    np.savetxt('test_predictions.csv', preds, fmt='%d', delimiter='\n')