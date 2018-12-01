from orderedattrdict import AttrDict
from codec import Codec

class NodeAttrsEncoder(Codec):
    def __init__(self, modelArgs):
        super().__init__(modelArgs)
        # Build inputs.
        nodeNamesTensorInput = Input(shape=(
            None,
            modelArgs.max_node_count,
            modelArgs.max_node_name_len,
            modelArgs.value_symbols_count,
            ))
        nodeAttrsTensorInput = Input(shape=(
            None,
            modelArgs.max_node_count,
            modelArgs.total_attrs_count,
            modelArgs.max_node_attr_len,
            modelArgs.value_symbols_count,
            ))
        nodeAdjacencyTensorInput = Input(shape=(
            None,
            modelArgs.max_node_count,
            modelArgs.max_node_fanout+1,
            ))
        self._inputs = [nodeNamesTensorInput, nodeAttrsTensorInput, nodeAdjacencyTensorInput]

        vectorisedNodeAttrsTensor = Dense()((nodeNamesTensorInput, nodeAttrsTensorInput))
        layers = [vectorisedNodeAttrsTensor]
        for i in range(modelArgs.graph_encoder_stack_depth):
            layers.append(self.__propagateNodeInfo(nodeAdjacencyTensorInput, layers[-1]))

        self._outputs = layers[-1]

    def __propagateNodeInfo(nodeAdjacencySpecTensor, nodeInfoTensor):
        nodeInfoUpdatedForNbrsTensor = Dense(nodeInfoTensor.shape[0])(nodeInfoTensor)
        parentNodeInfoTensor = IndexedNodeAverageLayer(
                nodeInfoUpdatedForParentTensor,
                nodeAdjacencySpecTensor[..., 0..1])
        nodeInfoUpdatedForParentTensor = Dense(nodeInfoTensor.shape[0])(nodeInfoTensor)
        nbrNodesInfoTensor = IndexedNodeAverageLayer(nodeInfoUpdatedForNbrsTensor, nodeAdjacencySpecTensor[..., 1...])

        updateGateTensor = Activation()
        resetGateTensor = Activation()
        parentGateTensor = Activation()

        newNodeInfoTensor = parentGateTensor * parentNodeInfoTensor + (1-parentGateTensor) * nbrNodesInfoTensor
        propagatedNodeInfoTensor = (1-updateGateTensor) * nodeInfoTensor + updateGateTensor * newNodeInfoTensor

        return propagatedNodeInfoTensor

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs

