from orderedattrdict import AttrDict
from codec import Codec

class ChildrenCodec(Codec):
    def __init__(self, modelArgs):
        super().__init__(modelArgs)

    def encoderInputs(self):
        # Build inputs.
        childrenTensorsInput = Input(shape=(None,
                         self.modelArgs.max_node_children_count,
                         self.modelArgs.total_schema_attributes_count))

        return AttrDict({"children": childrenTensorsInput})

    def encoder(self, inputs):
        _, childrencodes_h, childrencodes_c = GRU(modelArgs.children_code_dim, return_state=True)(inputs.children)
        childrenCode = (childrencodes_h, childrencodes_c)
        return childrenCode

    def decoder(self, attrbutesTensor, childrenTensor):
        # Instantiate decoder GRU.
        childrenDecoder = GRU(modelArgs.children_decoder_dim, return_sequences=True, return_state=True)

        # Apply decoder GRU on encoder outputs.
        childrenCode = childrenDecoder(decoderInputs)

        # Apply activation on children and attribute code.
        childrenCodeActivated = Dense(modelArgs.num_children_tokens, activation='softmax')

        return childrenCodeActivated


    def autoConfigXmls(self, xmlList):
        self.modelArgs.nodeNameDict = {}

    def autoConfigXmlNodes(self, xmlNodes):
        pass

    def save():
        pass

    def load():
        pass

    def decode(self, childNodes):
        pass
