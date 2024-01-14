import numpy as np
import math 
import random as rd

class NeuralNet:

    def __init__(self, X, Y, nodes, activations, task, X_test=None, Y_test=None):
        assert X.shape[1]==Y.shape[1], "X and Y don't have correct shapes!"
        assert len(nodes)==len(activations), "Mismatch between lengths of 'nodes' and 'activations'"
        assert task in ['binary', 'regression', 'multilabel', 'multiclass'], "Invalid task type given"
        for act in activations:
            assert act in ['relu', 'lrelu', 'sigmoid', 'tanh', 'softmax', 'linear'], "Invalid actiavtion given"
        self.X = X
        self.Y = Y
        self.nodes = nodes
        self.activations = activations
        self.task = task
        self.X_test = X_test
        self.X_test = Y_test
        self.layers = len(nodes)
        self.init_parameters()

    def init_parameters(self):
        num_nodes_mod = [self.X.shape[0]]
        num_nodes_mod.extend(self.nodes)
        total_param = 0
        self.init_param = {}
        for i in range(1, self.layers+1):
            #W = np.random.uniform(-1., 1., size=(num_nodes_mod[i], num_nodes_mod[i-1]))/np.sqrt(num_nodes_mod[i]*num_nodes_mod[i-1])
            W = np.random.randn(num_nodes_mod[i], num_nodes_mod[i-1])*0.01
            b = np.zeros((num_nodes_mod[i], 1))
            self.init_param['W'+str(i)] = W
            self.init_param['b'+str(i)] = b
            total_param += num_nodes_mod[i]*(num_nodes_mod[i-1]+1)
            print(f"W{i} shape: {W.shape}")
            print(f"b{i} shape: {b.shape}")
        print(f"Total Parameters: {total_param}")

    def relu(self, Z, derivative=False, d_A=0):
        if not derivative:
            A = Z*((Z > 0).astype(int))
            return A
        else:
            d_Z = d_A*((Z>0).astype(int))
            return d_Z
        
    def lrelu(self, Z, derivative=False, d_A=0):
        if not derivative:
            A = np.where(Z<0, 0.01*Z, Z)
            return A
        else:
            d_Z = ((Z>0).astype(int))
            d_Z = d_A*np.where(d_Z==0, 0.01, d_Z)
            return d_Z
        
    def sigmoid(self, Z, derivative=False, d_A=0):
        A = 1/(1 + np.exp(-Z))
        if not derivative:
            return A
        else:
            d_Z = d_A*A*(1-A)
            return d_Z
        
    def tanh(self, Z, derivative=False, d_A=0):
        A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        if not derivative:
            return A
        else:
            d_Z = d_A*(1-A*A)
            return d_Z
        
    def softmax(self, Z, derivative=False, d_A=0):
        A = np.exp(Z)/np.sum(np.exp(Z), axis=0)
        if not derivative:
            return A
        else:
            d_Z = A*(d_A - np.sum(d_A*A, axis=0))
            return d_Z
        
    def linear(self, Z, derivative=False, d_A=0):
        if not derivative:
            return Z
        else:
            return d_A
        
    def activation_fn(self, name, Z, derivative=False, d_A=0):
        out = eval(f"self.{name}(Z, derivative, d_A)")
        return out
        
    def forward_pass(self, param, X, return_cache=False):
        if X.ndim == 1: #happens while performing froward prop for a single examples
            X = X.reshape((len(X), 1))

        cache = {'A0': X}

        for i in range(1, self.layers+1):
            cache['Z'+str(i)] = param['W'+str(i)]@cache['A'+str(i-1)] + param['b'+str(i)]
            cache['A'+str(i)] = self.activation_fn(self.activations[i-1], cache['Z'+str(i)])
        if return_cache:
            return cache
        else:
            return cache['A'+str(self.layers)]
        
    def compute_performance(self, param, set, lambd, print_opt=True):
        #set --> 'train'/'test'
        if set=='train': X=self.X; Y=self.Y
        elif set=='test': X=self.X_test; Y=self.Y_test
        m = X.shape[1]
        weight_sum = 0
        for i in range(1, self.layers+1):
            W = param['W'+str(i)]
            weight_sum += np.sum(W**2)
        Y_hat = self.forward_pass(param, X)

        if self.task == 'binary':
            Y_hat_thresh = ((Y_hat>0.5).astype('int')) #apply threshold
            corr_incorr = (Y_hat_thresh == Y)
            correct = np.sum(corr_incorr)
            incorrect = m - correct
            accuracy = round((correct/m)*100, 3)
            if (Y_hat==0).any() or (Y_hat==1).any():
                cost = (-1/m)*np.sum(Y*np.log(Y_hat+1e-08) + (1-Y)*np.log(1-Y_hat+1e-08)) + (lambd/(2*m))*weight_sum
            else:
                cost = (-1/m)*np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)) + (lambd/(2*m))*weight_sum

        elif self.task == 'regression':
            cost = (1/(2*m))*np.sum((Y-Y_hat)**2, axis=1) + (lambd/(2*m))*weight_sum
            accuracy = '--'

        elif self.task == 'multiclass':
            if (Y_hat==0).any():
                cost = (-1/m)*np.sum(Y*np.log(Y_hat+1e-08)) + (lambd/(2*m))*weight_sum
            else:
                cost = (-1/m)*np.sum(Y*np.log(Y_hat)) + (lambd/(2*m))*weight_sum

            max_indices = np.argmax(Y_hat, axis=0)
            filtered = np.zeros_like(Y_hat)
            rows = np.arange(Y_hat.shape[1])
            filtered[max_indices, rows] = 1
            correct = np.sum(np.all(filtered==Y, axis=0).astype('int'))
            incorrect = m-correct
            accuracy = round((correct/m)*100, 3)

        elif self.task == 'multilabel':
            if (Y_hat==0).any() or (Y_hat==1).any():
                cost = (-1/m)*np.sum(Y*np.log(Y_hat+1e-08) + (1-Y)*np.log(1-Y_hat+1e-08)) + (lambd/(2*m))*weight_sum
            else:
                cost = (-1/m)*np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)) + (lambd/(2*m))*weight_sum

            Y_hat_thresh = ((Y_hat>0.5).astype('int')) #apply threshold
            correct = np.sum(np.all(Y_hat_thresh==Y, axis=0).astype('int'))
            incorrect = m-correct
            accuracy = round((correct/m)*100, 3)

        if print_opt:
            if self.task != 'regression':
                print(f"Correct: {correct} | Incorrect: {incorrect}")
            print(f"Accuracy: {accuracy} | Cost: {cost}")
        else:
            return (accuracy, cost)

    def return_seed(self, A_L, Y):
        m = Y.shape[1]
        if self.task == 'binary' or self.task == 'multilabel':
            if (A_L==0).any() or (A_L==1).any():
                seed = (1/m)*((1/(1-A_L+1e-08))*(1-Y) - (1/(A_L+1e-08))*Y)
            else:
                seed = (1/m)*((1/(1-A_L))*(1-Y) - (1/A_L)*Y)
        elif self.task == 'multiclass':
            if (A_L==0).any():
                seed = (-1/m)*(Y/(A_L+1e-08))
            else:
                 seed = (-1/m)*(Y/A_L)
        elif self.task == 'regression':
            seed = (1/m)*(A_L-Y)
        return seed
        
    def backward_pass(self, param, X, Y, lambd):
        if X.ndim == 1: X = X.reshape((len(X), 1))
        if Y.ndim == 1: Y = Y.reshape((len(Y), 1))

        forward_cache = self.forward_pass(param, X, return_cache=True)
        derivatives = {}
        m = X.shape[1]
        A_L = forward_cache['A'+str(self.layers)]
        d_A = self.return_seed(A_L, Y)
        for i in range(self.layers, 0, -1):
            Z = forward_cache['Z'+str(i)]
            A_prev = forward_cache['A'+str(i-1)]
            W = param['W'+str(i)]

            d_Z = self.activation_fn(self.activations[i-1], Z, derivative=True, d_A=d_A)
            
            d_W = d_Z@(A_prev.T)
            d_b = np.sum(d_Z, axis=1, keepdims=True)
            d_A = (W.T)@d_Z

            derivatives['d_W'+str(i)] = d_W + (lambd/m)*W #since we are regularizing weights
            derivatives['d_b'+str(i)] = d_b
        return derivatives
    
    def train(self, epochs, batch_size, alpha, beta1=0.9, beta2=0.999, E=1e-08, lambd=0, decay_rate=0):
        param = self.init_param
        if batch_size == -1:
            X_batch = self.X
            Y_batch = self.Y

        cost_hist = []
        if self.task != 'regression': accuracy_hist = []
        accuracy, cost = self.compute_performance(param, 'train', lambd, print_opt=False)
        cost_hist.append(cost)
        if self.task != 'regression': accuracy_hist.append(accuracy)
        print(f"At epoch 0 --> Accuracy: {accuracy} | Cost: {cost}")

        train_param = param.copy()
        adam_cache = {}
        for i in range(1, self.layers+1): #initialize adam_cache
            adam_cache['V_W'+str(i)] = 0
            adam_cache['V_b'+str(i)] = 0
            adam_cache['S_W'+str(i)] = 0
            adam_cache['S_b'+str(i)] = 0

        for epoch in range(1, epochs+1):
            alpha_ = alpha/(1+(i*decay_rate))
            if batch_size != -1:
                batch_indices = rd.choices(list(range(self.X.shape[1])), k=batch_size)
                X_batch = self.X[:, batch_indices]
                Y_batch = self.Y[:, batch_indices]

            derivatives = self.backward_pass(train_param, X_batch, Y_batch, lambd)

            for i in range(self.layers, 0, -1):
                d_W = derivatives['d_W'+str(i)]
                d_b = derivatives['d_b'+str(i)]
                W = train_param['W'+str(i)]
                b = train_param['b'+str(i)]

                V_W = adam_cache['V_W'+str(i)]
                V_b = adam_cache['V_b'+str(i)]
                S_W = adam_cache['S_W'+str(i)]
                S_b = adam_cache['S_b'+str(i)]

                V_W = beta1*V_W + (1-beta1)*d_W
                V_b = beta1*V_b + (1-beta1)*d_b
                S_W = beta2*S_W + (1-beta2)*(d_W**2)
                S_b = beta2*S_b + (1-beta2)*(d_b**2)
                V_W_c, V_b_c = V_W/(1-beta1**epoch), V_b/(1-beta1**epoch)
                S_W_c, S_b_c = S_W/(1-beta2**epoch), S_b/(1-beta2**epoch)

                #update the parameters of ith layer
                W = W - ((alpha_/(np.sqrt(S_W_c)+E))*V_W_c)
                b = b - ((alpha_/(np.sqrt(S_b_c)+E))*V_b_c)

                #writing values back to the respective caches
                train_param['W'+str(i)] = W
                train_param['b'+str(i)] = b

                adam_cache['V_W'+str(i)] = V_W
                adam_cache['V_b'+str(i)] = V_b
                adam_cache['S_W'+str(i)] = S_W
                adam_cache['S_b'+str(i)] = S_b

            accuracy, cost = self.compute_performance(train_param, 'train',lambd, print_opt=False)
            cost_hist.append(cost)
            if self.task != 'regression': accuracy_hist.append(accuracy)

            if epoch% math.ceil(epochs/10) == 0 or epoch == (epochs):
                print(f"At epoch {epoch} --> Accuracy: {accuracy} | Cost: {cost}")

        self.trained_param = train_param
        if self.task != 'regression': self.accuracy_hist = accuracy_hist
        self.cost_hist = cost_hist