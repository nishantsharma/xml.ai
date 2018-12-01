from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class random_initializer(object):
    """
    To initialize a tensor with random values such that:
    a) L2 Norm of each sub-tensor along input axes is 1.
    b) Matrix is sampled from normal distribution. That means that each individual
       sub-tensor is picked in a spherically symmetrical manner.
    """
    def __init__(self, axis):
        self.axis = tuple(axis)

    def get_config(self):
        return { "axis" : self.axis }

    def __call__(self, shape, dtype=None):
        retval = np.random.normal(size=shape)
        retval /= np.sqrt(np.sum(retval*retval, axis=self.axis, keepdims=True))
        return retval

def random_init_unit_norm(shape, dtype=None):
    """
    To initialize a tensor with random values such that:
    a) L2 Norm for entire matrix is 1.
    b) Data is sampled from normal distribution. That means that each individual
       sub-tensor is picked in a spherically symmetrical manner.
    """
    retval = np.random.normal(size=shape)
    retval /= np.linalg.norm(retval)
    return retval

class NodeInfoPropagateLayer(Layer):
    """
    A layer to propagate information at individual nodes, across the network.
    """
    def __init__(self, wdith, **kwargs):
        self.width = width
        super(TypeCastLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["width"] = self.width
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      trainable=True,
                                      initializer=self.weight_initializer,
                                      constraint=self.weight_constraint)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, nodeAdjacencySpecTensor, nodeInfoTensor):
        parentInfoTensor = IndexedAverageOp(nodeInfoAdjacency[..., 0..1], nodeInfoTensor)
        nbrNodesInfoTensor = IndexedAverageOp(nodeInfoAdjacency[..., 1:], nodeInfoTensor)
        nodeInfoPropagatedTensor = Tensor(nodeInfoTensor.shape())
        # Create resetGateTensor, updateGateTensor

        return K.cast(x, self.to_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

class MatrixProductLayer(Layer):
    """
    A keras layer to multiply the input matrix with a trainable weight kernel.
    """
    def __init__(self,
                 output_dim,
                 weight_constraint=None,
                 weight_initializer=random_init_unit_norm,
                 **kwargs):
        self.output_dim = output_dim
        self.weight_constraint = weight_constraint
        self.weight_initializer = weight_initializer
        super(MatrixProductLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        # config["weight_constraint"] = self.weight_constraint
        # config["weight_initializer"] = self.weight_initializer
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      trainable=True,
                                      initializer=self.weight_initializer,
                                      constraint=self.weight_constraint)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

