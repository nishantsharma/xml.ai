from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NodeInfoPropagate(nn.Module):
    def __init__(self, modelArgs):
        super().__init__()
        self.modelArgs = modelArgs

        self.parentOp = torch.nn.FullyConnected()
        self.neighborOp = torch.nn.FullyConnected()

        # Neighbor info gate.
        self.gruCell = torch.nn.GRUCell()

    def forward(nodeAdjacencySpecTensor, nodeNamesEncoded, nodeAttributesEncoded):
        nodeInfosEncoded = self.concat(nodeNamesEncoded, nodeAttributesEncoded)

        nodeInfoPropagated = self.fullyConnected(nodeInfosEncoded)
        
        for i in range(modelArgs.graph_encoder_stack_depth):
            # Pre-compute node infos to propagate in the role of parent and neighbor respetivaly.
            parentInfosToPropagate = self.parentOp(nodeInfoPropagated)
            neighborInfosToPropagate = self.neighborOp(nodeInfoPropagated)

            # Compute the propagated node infos in loop.
            neighborsNodeInfoSummary = Tensor(nodeInfoPropagated.shape)
            for i, nodeAdjacencySpecTensorRow in enumerate(nodeAdjacencySpecTensor):
                parentInfoPropagatedRow = parentInfosToPropagate[nodeAdjacencySpecTensor[i][0]]
                neighborInfosPropagatedRow = Tensor(nodeInfoPropagated.shape[1:])
                nbrCount = 0
                for nbrIndex in nodeAdjacencySpecTensor[i][1:]:
                    if nbrIndex < 0:
                        break
                    neighborInfosPropagatedRow += neighborInfosToPropagate[nbrIndex]
                    nbrCount += 1
                neighborsNodeInfoSummary[i] = parentInfoPropagatedRow + neighborInfosPropagatedRow / nbrCount

            nodeInfoPropagated = self.gruCell(nodeInfoPropagated, neighborsNodeInfoSummary)

        return nodeInfoPropagated

