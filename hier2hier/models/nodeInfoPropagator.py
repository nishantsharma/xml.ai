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

        # Upgrade size of input.
        self.resizeInput = nn.Linear(self.encoded_node_vec_len, self.propagated_info_len)
        self.parentOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)
        self.neighborOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)

        # Neighbor info gate.
        self.gruCell = torch.nn.GRUCell(propagated_info_len, propagated_info_len)

    def forward(self, treeIndex2NodeIndex2NbrIndices, nodeInfosTensor):
        nodeInfoPropagated = self.resizeInput(nodeInfosTensor)
        
        for i in range(self.node_info_propagator_stack_depth):
            parentInfosToPropagate = torch.zeros(nodeInfoPropagated.shape)
            neighborInfosToPropagate = torch.zeros(nodeInfoPropagated.shape)

            # Propagate parent info into parentInfosToPropagate and neighborInfosToPropagate.
            for treeIndex, nodeIndex2NbrIndices in treeIndex2NodeIndex2NbrIndices.items():
                for nodeIndex, (parentIndex, childIndices) in nodeIndex2NbrIndices.items():
                    parentInfosToPropagate[treeIndex, nodeIndex] = nodeInfoPropagated[treeIndex, parentIndex]
                    for childIndex in childIndices:
                        neighborInfosToPropagate[treeIndex, nodeIndex] += nodeInfoPropagated[treeIndex, childIndex]
                    neighborInfosToPropagate[treeIndex, nodeIndex] /= len(childIndices)

            # Pre-compute node infos to propagate in the role of parent and neighbor respetivaly.
            parentInfosToPropagate = self.parentOp(parentInfosToPropagate)
            neighborInfosToPropagate = self.neighborOp(neighborInfosToPropagate)

            # Compute final neighbor information summary.
            neighborsNodeInfoSummary = parentInfosToPropagate + neighborInfosToPropagate

            # Pass the new information through GRU cell to obtain updated node info.
            for sampleIndex in range(nodeInfoPropagated.shape[0]):
                nodeInfoPropagated[sampleIndex] = self.gruCell(nodeInfoPropagated[sampleIndex], neighborsNodeInfoSummary[sampleIndex])

        return nodeInfoPropagated

