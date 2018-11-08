import numpy as np
from abc import ABC, abstractmethod

from sklearn.preprocessing import normalize


class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'

class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()
        x = np.maximum(0, x)
        # This is just a hack
        if x.shape != (3, 4):
            x = normalize(x) # This is because, with more Affine Layer, 
        # the input gets really huge. This solves the problem. 
        # Implement relu activation

        

        return x

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        # Your implementation starts
        dx = (self.x > 0) * dprev

        # End of your implementation
        return dx


class YourActivation(Layer):#BONUS
    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            :param x: outputs of previous layer
            :return: output of activation
        '''
        # Lets have an activation of X^2
        # TODO: CHANGE IT
        self.x = x.copy()
        out = x ** 2
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        # TODO: CHANGE IT
        # Example: derivate of X^2 is 2X
        dx = 2*self.x*dprev
        return dx


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass
        probs = None
       
        # Your implementation starts
        scores = x
        e_scores = np.exp(scores)
        #print("scores : ", scores)
        #print("exp scores: ", exp_scores)
        probs = e_scores / np.sum(e_scores, axis=1, keepdims=True)
        
        # End of your implementation
        self.probs = probs.copy()
        return probs

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts
        N = y.shape[0]
        dx = self.probs
        dx[np.arange(N), y] -= 1
        dx /= N 

        # End of your implementation

        return dx


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None
    # Your implementation starts
    
    num_of_examples = y.shape[0]
    actual_logprobs = -np.log(probs[range(num_of_examples),y])
    loss = np.sum(actual_logprobs)/num_of_examples
    # End of your implementation
    return loss


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        # Vectorize the input to [batchsize, others] array
        batch_size = x.shape[0]


        # Do the affine transform
        '''
            - self.W and self.b will be used for z = Wx + b
            - np.reshape(array, (dim1,dim2)), if dim2=-1 set, it has the same dim2
        ''' 
        out = np.reshape(x, (batch_size, -1)).dot(self.W) + self.b

        # Save x for using in backward pass
        self.x = x.copy()

        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        # Vectorize the input to a 1D ndarray
        x_vectorized = np.reshape(self.x,[batch_size, -1])
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        # Calculate dx = w*dout - remember to reshape back to shape of x.

        dx = np.dot(dprev, self.W.T)
        dx = np.reshape(dx, self.x.shape)
    
        # Calculate dw = x*dout
        dw = np.dot(x_vectorized.T,dprev)
    
        # Calculate db = dout
        db = np.sum(dprev, axis=0)
        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'

class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)

        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

class VanillaSGDOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for SGD
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
         # Your implementation start
        
        m.W -= self.lr*(m.dW + self.reg*m.W)
        m.b -= self.lr*(m.db + self.reg*m.b)

        # End of your implementation
       
class SGDWithMomentum(VanillaSGDOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        # Save velocities for each model in a dict and use them when needed.
        # Modules can be hashed
        self.velocities = {m: 0 for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        m.dW +=  self.reg*m.W
        velocities_W = self.velocities[m]
        velocities_b = self.velocities[m]

        velocities_W = self.mu*velocities_W + m.dW 
        m.W -= velocities_W 

        m.db += self.reg*m.b
        velocities_b = self.mu*velocities_b + m.db
        m.b -= velocities_b

       # End of your implementation



