from .layer import Layer
from copy import copy
from abc import abstractmethod
import numpy as np


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


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        # Vectorize the input to a 1D ndarray
        batch_size = x.shape[0]
        x_vectorized = np.reshape(x, [batch_size, -1])

        # Do the affine transform

        # Use your past implementation if needed

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
        x_vectorized = np.reshape(self.x, [batch_size, -1])
        dx, dw, db = None, None, None

        # YOUR CODE STARTS

        # Use your past implementation if needed

        # YOUR CODE ENDS
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'


class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        # Calculate output's H and W according to your lecture notes
        out_H = np.int(((H + 2*self.padding - FH) / self.stride) + 1)
        out_W = np.int(((W + 2*self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])
        # TO DO: Do cross-correlation by usng for loops
        for n in range(N): # This is for iterating over the inputs 
            for f in range(F): # This is for iterating over the kernels/filters
                for h in range(out_H):
                    for w in range(out_W):
                        # e_w_p is the striding part. 
                        element_wise_product = np.multiply(padded_x[n, :, h*self.stride:h*self.stride+FH, 
                            w*self.stride:w*self.stride+FW], self.W[f, :])
                        out[n, f, h, w] = np.sum(element_wise_product) + self.b[f]
        return out

    def backward(self, dprev):
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)

       # db = None
       # dw = None
       # dx = None

        # Your implementation here
        for n in range(N): # This is for iterating over the inputs 
            for f in range(F): # This is for iterating over the kernels/filters
                db[f] += np.sum(dprev[n, f])
                for h in range(out_H):
                    for w in range(out_W):
                        dw[f] += np.multiply(padded_x[n, :, h*self.stride:h*self.stride+FH, 
                            w*self.stride:w*self.stride+FW], dprev[n, f, h, w])
                        rotated_W = np.rot90(self.W[f, :], 2)
                        dx_temp[n, :, h*self.stride:h*self.stride+FH, 
                            w*self.stride:w*self.stride+FW] += np.multiply(self.W[f],
                                    dprev[n, f, h, w])
        dx = dx_temp[:, :, self.padding:self.padding + H, self.padding:self.padding+W]
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db
