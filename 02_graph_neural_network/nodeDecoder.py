from orderedattrdict import AttrDict
from codec import Codec

class NodeDecoder(Codec):
    def __init__(self, modelArgs):
        super().__init__(modelArgs)
        # Build inputs.
        nodeInfoCodeInput = Input(shape=(
            None,
            modelArgs.max_node_count,
            modelArgs.node_info_vec_len,
            ))
        nodeAdjacencyInput = Input(shape=(
            None,
            modelArgs.max_node_count,
            modelArgs.max_node_fanout+1,
            ))
        self._inputs = [nodeInfoCodeInput, nodeAdjacencyInput]

        # Define shared layers as global variables.
        repeator = RepeatVector(nodeInfoCodeInput)
        concatenator = Concatenate(axis=-1)
        densor = Dense(1, activation="relu")
        activator = Activation(softmax, name='attention_weights')
        dotor = Dot(axis=1)

        # Method to compute the context after applying attention for one output.
        def getContextWithAttention(nodeDecoderStatePrev):
            nodeDecoderStatePrev = repeator(nodeDecoderStatePrev)
            nodeAttentionInputs = concatenator(nodeInfoCodeInput, nodeDecoderStatePrev)
            nodeAttentionEnergies = densor(nodeAttentionInputs)
            nodeAttentionAlphas = activator(nodeAttentionEnergies)
            nodeDecoderContext = dotor(nodeAttentionAlphas, nodeInfoCodeInput)
            return nodeDecoderContext

        nodeDecoderGruCell = GRU(n_s, return_state = True)

        outputs = []
        nodeDecoderStatePrev = Input(shape=(modelArgs.node_decoder_gru_width,))
        for t in range(modelArgs.max_output_len)
            # Compute decoder context for current cell.
            nodeDecoderContext = getContextWithAttention(nodeDecoderStatePrev)
            nodeDecoderStatePrev, _, = nodeDecoderGruCell(
                    nodeDecoderContext,
                    initial_state = nodeDecoderStatePrev,
                    )
            out = output_layer(nodeDecoderStatePrev)
            outputs.append(out)
        nodeInfoToAttend = Dot(nodeInfoCodeInput, nodeAdjacencyInput)

        decodedSignalsTensor = Concat(outputs)

        # Build outputs.
        decodedValueTensor = Dense(
                modelArgs.value_symbols_count,
                activation="SIGMOID")(decodedSignalsTensor)
        self._outputs = [decodedValueTensor]

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs

