from codec import Codec

class ChildrenCodec(Codec):
    def __init__(self):
        super().__init__() 
    def encoder():
        childTensors = childrenCodec.tensorInputs()
        childrenCodeTensor = GRU(modelArgs.children_code_dim, return_state=False)(childTensors)
    def setupUsingXmls(self, xmlList):
        pass
    def setupUsingXmlNodes(self, xmlNodes):
        pass
    def save():
        pass
    def load():
        pass
    def encode(self, childNodes):
        """
        Each node is either a child path or full XML node text.
        """
        for childNode in childNodes:
            if isinstance(childNode, str):
                # This is the path to child.
                pass
            elif isinstance(childNode, dict):
                # This is an explicit new child.
                pass
        pass
    def decode(self, childNodes):
        pass
