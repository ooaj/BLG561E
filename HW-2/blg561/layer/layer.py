
import numpy as np
from abc import ABC, abstractmethod
from .helpers import flatten_unflatten


class Layer(ABC):
    def __init__(self, input_size, output_size):
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
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        #  This is for preventing reference issues in numpy arrays
        x = x.copy()
        # Use your past implementation if needed

        return x

    def backward(self, dprev):
        dx = None
        # Use your past implementation if needed

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
        softmax_scores = None

        # Use your past implementation if needed

        self.probs = softmax_scores.copy()
        return softmax_scores

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts

        # Use your past implementation if needed

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

    # Use your past implementation if needed

    # End of your implementation
    return loss


class Dropout(Layer):
    def __init__(self, p=.5):
        '''
            :param p: dropout factor
        '''
        self.mask = None
        self.mode = 'train'
        self.p = p

    def forward(self, x, seed=None):
        '''
            :param x: input to dropout layer
            :param seed: seed (used for testing purposes)
        '''
        if seed is not None:
            np.random.seed(seed)
        # YOUR CODE STARTS
        if self.mode == 'train':

            # Create a dropout mask
            mask = (np.random.rand(*x.shape) < self.p) / self.p 
            # Do not forget to save the created mask for dropout in order to use it in backward
            self.mask = mask.copy()
            out = x*mask
            return out
        elif self.mode == 'test':
            out = x
            return out
        # YOUR CODE ENDS
        else:
            raise ValueError('Invalid argument!')

    def backward(self, dprev):

        dx = dprev*self.mask
        return dx


class BatchNorm(Layer):
    def __init__(self, D, momentum=.9):
        self.mode = 'train'
        self.normalized = None

        self.x_sub_mean = None
        self.momentum = momentum
        self.D = D
        self.running_mean = np.zeros(D)
        self.running_var = np.zeros(D)
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.ivar = np.zeros(D)
        self.sqrtvar = np.zeros(D)

    # @flatten_unflatten
    def forward(self, x, gamma=None, beta=None):
        if self.mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            if gamma is not None:
                self.gamma = gamma.copy()
            if beta is not None:

                self.beta = beta.copy()

            # Normalise our batch
            self.normalized = ((x - sample_mean) /
                               np.sqrt(sample_var + 1e-5)).copy()
            self.x_sub_mean = x - sample_mean

            # YOUR CODE HERE

            # Update our running mean and variance then store.

            running_mean = None
            running_var = None

            out = None

            # YOUR CODE ENDS
            self.running_mean = running_mean.copy()
            self.running_var = running_var.copy()

            self.ivar = 1./np.sqrt(sample_var + 1e-5)
            self.sqrtvar = np.sqrt(sample_var + 1e-5)

            return out
        elif self.mode == 'test':
            out = None
        else:
            raise Exception(
                "INVALID MODE! Mode should be either test or train")
        return out

    def backward(self, dprev):
        N, D = dprev.shape
        # YOUR CODE HERE
        dx, dgamma, dbeta = None, None, None
        # Calculate the gradients
        return dx, dgamma, dbeta


class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = np.int(((H - self.pool_height) / self.stride) + 1)
        out_W = np.int(((W - self.pool_width) / self.stride) + 1)
        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])

        # Implement MaxPool
        # YOUR CODE HERE
        for n in range(N): # Iterate over the input
            for h in range(out_H): 
                for w in range(out_W):
                    h_inc = h*self.stride
                    w_inc = w*self.stride
                    temp_x = x[n, :, h_inc:h_inc + self.pool_height, w_inc:w_inc + self.pool_width]
                    temp_out = np.zeros(out_H)
                    for counter, submatrix in enumerate(temp_x):
                        temp_out[counter] = np.max(submatrix)
                    out[n, :, h, w] = temp_out

        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)

        # Calculate the gradient (dx)
        # YOUR CODE HERE
        return dx


class Flatten(Layer):
    def __init__(self):
        self.N, self.C, self.H, self.W = 0, 0, 0, 0

    def forward(self, x):
        self.N, self.C, self.H, self.W = x.shape
        out = x.reshape(self.N, -1)
        return out

    def backward(self, dprev):
        return dprev.reshape(self.N, self.C, self.H, self.W)
