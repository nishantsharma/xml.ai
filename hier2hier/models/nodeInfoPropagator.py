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


class NodeInfoPropagator(nn.Module):
    def __init__(self,
            encoded_node_vec_len,
            propagated_info_len,
            node_info_propagator_stack_depth):
        super().__init__()
        self.encoded_node_vec_len = encoded_node_vec_len
        self.propagated_info_len = propagated_info_len
        self.node_info_propagator_stack_depth = node_info_propagator_stack_depth

        self.parentOp = torch.nn.ReLU()
        self.neighborOp = torch.nn.ReLU()

        # Neighbor info gate.
        self.gruCell = torch.nn.GRUCell(encoded_node_vec_len, propagated_info_len)

    def forward(nodeAdjacencySpecTensor, nodeInfosEncoded):
        nodeInfoPropagated = self.fullyConnected(nodeInfosEncoded)
        
        for i in range(self.node_info_propagator_stack_depth):
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

