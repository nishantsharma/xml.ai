from orderedattrdict import AttrDict
from codec import Codec

class AttributesCodec(Codec):
    def __init__(self, modelArgs):
        super().__init__(modelArgs)

    def encoderInputs(self):
        # Build inputs.
        nodeNameCharTensorsInput = Input(shape=(None, modelArgs.schemaNodeCount))
        parentInput = Input(shape=(None, modelArgs.attrs_code_dim))
        numChildrenInput = Input(shape=(None, 1))
        attrIdTensorsInput = Input(shape=(None,
            self.modelArgs.max_node_attrbutes_count,
            self.modelArgs.total_schema_attributes_count))
        attrValueTensorsInput = Input(shape=(None,
            self.modelArgs.max_node_attrbutes_count,
            self.modelArgs.max_attribute_values_len))

        # Return inputs list.
        allInputs = AttrDict({
                "nodeName": nodeNameCharTensorsInput,
                "parent": parentInput,
                "numChilren": numChildrenInput,
                "attrIds": attrIdTensorsInput,
                "attrValues": attrValueTensorsInput
                })
        return allInputs

    def encoder(self, inputs):
        modelArgs = self.modelArgs

        # Dereference list
        nodeNameCharTensorsInput = inputs[0]
        parentInput = inputs[1]
        numChildrenInput = inputs[2]
        attrIdTensorsInput = inputs[3]
        attrValueTensorsInput = inputs[4]

        # Encode node name and attributes.
        nodeNameTensor = GRU(modelArgs.name_code_dim, return_state=True)(inputs.nodeName)
        attrKeyValuePairTensors = Concat(attrIdtensorsInput, attrValueTensorsInput)
        attributesTensor = GRU(modelArgs.attrs_code_dim, return_state=True)(attrKeyValuePairTensors)
        attrEncoderInputs = Concat(nodeNameTensor, childCountTensor, attributesTensor)
        attrCodeTensor = GRU(modelArgs.attrs_code_dim, return_state=True)(attrEncoderInputs)

        # Concat initial state inputs.
        inputTensor = Concat(nodeNameInput, parentInput, numChildrenInput)

        # Reshaping inputs with a RELU layer.
        reshapedInputTensor = Dense(
                modelArgs.attrs_code_dim,
                activation="relu")(inputTensor)

        _, attrCodes_h, attrCodes_c = GRU(modelArgs.attrs_code_dim, return_state=True)(reshapedInputTensor)

        # Concatenate state outputs. That is the encoded state.
        attrCodes = [attrCodes_h, attrCodes_c]

        # Return the output.
        return attrCodes

    def encode(self, origNodeName, origChildren, origAttrs):
        pass

    def decoder(self, attrbutesTensor, childrenTensor):
        # Instantiate decoder GRU.
        attrsDecoder = GRU(modelArgs.attrs_decoder_dim, return_sequences=True, return_state=True)

        # Apply decoder GRU on encoder outputs.
        attrCode = attributeDecoder(decoderInputs)

        # Apply activation on children and attribute code.
        attrsCodeActivated = Dense(modelArgs.num_attrs_tokens, activation='softmax')

    def autoConfigXmls(self, xmlList):
        self.modelArgs.nodeNameDict = {}

    def autoConfigXmlNodes(self, xmlNodes):
        pass
    def save():
        pass
    def load():
        pass
    def encode(self, nodeName, attrList):
        pass
    def decode(self, attrList):
        pass
