from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

def propagateNodeInfo(nodeAdjacencySpecTensor, nodeInfoTensor):
    nodeInfoUpdatedForNbrsTensor = Dense(nodeInfoTensor.shape[0])(nodeInfoTensor)
    parentNodeInfoTensor = IndexedNodeAverageLayer(nodeInfoUpdatedForParentTensor, nodeAdjacencySpecTensor[..., 0..1])

    nodeInfoUpdatedForParentTensor = Dense(nodeInfoTensor.shape[0])(nodeInfoTensor)
    nbrNodesInfoTensor = IndexedNodeAverageLayer(nodeInfoUpdatedForNbrsTensor, nodeAdjacencySpecTensor[..., 1...])

    updateGateTensor = Activation()
    resetGateTensor = Activation()
    parentGateTensor = Activation()

    newNodeInfoTensor = parentGateTensor * parentNodeInfoTensor + (1-parentGateTensor) * nbrNodesInfoTensor
    propagatedNodeInfoTensor = (1-updateGateTensor) * nodeInfoTensor + updateGateTensor * newNodeInfoTensor

    return propagatedNodeInfoTensor


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

