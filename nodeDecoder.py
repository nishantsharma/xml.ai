from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NodeDecoder(nn.Module):
    START_SYMBOL = Tensor(0)
    def __init__(self, modelArgs):
        super().__init__()
        self.modelArgs = modelArgs

        self.GruCell = nn.GruCell()
        self.symbolDecoder = nn.Softmax()
        self.attentionDecoder = nn.Softmax()

    def forward(self, nodeInfoPropagated, nodeAdjacencyInput):
        # Build inputs.
        curSymbol = START_SYMBOL
        curAttention = Tensor([1 if nodeAdj[0] == i else 0 for i, nodeAdj in enumerate(nodeAdjacencyInput)])

        # Recurrent loop.
        gruCurState = nn.zeros()
        outputs = []
        while True:
            nodeInfoToAttend = Dot(nodeInfoPropagated, curAttention)
            gruInput = Concat(nodeInfoToAttend, curSymbol)
            gruState = self.GruCell(gruState, gruInput)

            # Next compute next symbol and attention.
            curSymbol = self.symbolDecoder(gruState)
            curAttention = self.attentionDecoder(gruState, nodeInfosPropagated)
            if curSymbol != END_SYMBOL:
                outputs.append(curSymbol)
            else:
                break
                
        return outputs