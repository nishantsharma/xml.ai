from __future__ import unicode_literals, print_function, division
from io import open
from collections import OrderedDict
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch import optim
import torch.nn.functional as F
from hier2hier.util import (onehotencode, checkNans, invertPermutation, blockProfiler,
                            methodProfiler, lastCallProfile)
from hier2hier.models.moduleBase import ModuleBase

class HierarchyPropagator(ModuleBase):
    def __init__(self,
            propagated_info_len,
            node_info_propagator_stack_depth,
            disable_batch_norm,
            input_dropout_p=0,
            dropout_p=0,
            device=None):
        super().__init__(device)
        self.propagated_info_len = propagated_info_len
        self.node_info_propagator_stack_depth = node_info_propagator_stack_depth
        self.disable_batch_norm = disable_batch_norm

        self.input_dropout = nn.Dropout(p=input_dropout_p)

        # Batch norm on propagated info.
        if not self.disable_batch_norm:
            self.batchNormPropagatedInfo = nn.BatchNorm1d(num_features=propagated_info_len)

        # Parent info gate.
        self.overlayParentInfo = torch.nn.GRUCell(propagated_info_len, propagated_info_len)

        # Children info gate.
        self.overlayChildrenInfo = torch.nn.GRUCell(propagated_info_len, propagated_info_len)

        # GRU output dropout.
        self.dropout = nn.Dropout(p=dropout_p)

    def reset_parameters(self, device):
        if not self.disable_batch_norm:
            self.batchNormPropagatedInfo.reset_parameters()
        self.overlayParentInfo.reset_parameters()
        self.overlayChildrenInfo.reset_parameters()

    @methodProfiler
    def forward(self,
            nodeInfoPropagatedByDfo,
            parentsSelectorByDFO,
            childrenSelectorListByDFO,
            decreasingFanoutsFactor,
            tensorBoardHook):
        # Apply input dropout. Better to do after resize as input bits are non-uniform.
        nodeInfoPropagatedByDfo = self.input_dropout(nodeInfoPropagatedByDfo)

        # Apply batch norm.
        if not self.disable_batch_norm:
            nodeInfoPropagatedByDfo = self.batchNormPropagatedInfo(nodeInfoPropagatedByDfo)

        # Run propagation loop.
        for i in range(self.node_info_propagator_stack_depth):
            # Prepare parent info for propagation into new nodeInfoPropagated.
            parentInfoPropagatedByDfo = nodeInfoPropagatedByDfo[parentsSelectorByDFO, ...]

            # Compute children info to propagate to each node.
            childrenInfoPropagatedByDfo = torch.tensor([], device=self.device)
            for childrenSelectorByDFO in childrenSelectorListByDFO:
                curChildInfoPropagatedByDfo = nodeInfoPropagatedByDfo[childrenSelectorByDFO, ...]
                if not childrenInfoPropagatedByDfo.shape[0]:
                    # First iteration of the loop.
                    childrenInfoPropagatedByDfo = curChildInfoPropagatedByDfo
                else:
                    assert(curChildInfoPropagatedByDfo.shape[0] >= childrenInfoPropagatedByDfo.shape[0])
                    # If the fanout increases in current iteration, pad neighbor infos by the deficit.
                    if curChildInfoPropagatedByDfo.shape[0] > childrenInfoPropagatedByDfo.shape[0]:
                        deficit = curChildInfoPropagatedByDfo.shape[0] - childrenInfoPropagatedByDfo.shape[0]
                        childrenInfoPropagatedByDfo = nn.ZeroPad2d((0, 0, 0, deficit))(childrenInfoPropagatedByDfo)
                    childrenInfoPropagatedByDfo = childrenInfoPropagatedByDfo + curChildInfoPropagatedByDfo

            # Propagate the parent information using GRU cell to obtain updated node info.
            nodeInfoPropagatedByDfo = self.overlayParentInfo(nodeInfoPropagatedByDfo, parentInfoPropagatedByDfo)

            if childrenInfoPropagatedByDfo.shape[0]:
                # Normalize childrenInfoPropagatedByDfo by child count.
                decreasingFanoutsFactor = decreasingFanoutsFactor.view(
                    list(decreasingFanoutsFactor.shape) + [1]
                )
                childrenInfoPropagatedByDfo = childrenInfoPropagatedByDfo / decreasingFanoutsFactor

            # There may still be some rows that do not have children.
            deficitStart = childrenInfoPropagatedByDfo.shape[0]
            finalDeficit = nodeInfoPropagatedByDfo.shape[0] - deficitStart
            
            if finalDeficit:
                # Keep original info, when no children.                    
                nodeInfoPropgatedByDfoWhenNoChild = nodeInfoPropagatedByDfo[deficitStart:]

            # Propagate children information using GRU cell to obtain updated node info.
            nodeInfoPropagatedByDfo = self.overlayChildrenInfo(
                nodeInfoPropagatedByDfo[0:deficitStart],
                childrenInfoPropagatedByDfo,
            )

            if finalDeficit:
                nodeInfoPropagatedByDfo = torch.cat([
                    # Propagate children information using GRU cell to obtain updated node info.
                    nodeInfoPropagatedByDfo,
                    nodeInfoPropgatedByDfoWhenNoChild,
                ])

            # Apply GRU output dropout.
            nodeInfoPropagatedByDfo = self.dropout(nodeInfoPropagatedByDfo)

        return nodeInfoPropagatedByDfo


if __name__ == "__main__":
    encoded_node_vec_len=1
    propagated_info_len=1
    max_node_count=9
    node_info_propagator_stack_depth=1
    treeIndex2NodeIndex2NbrIndices = [
            [ # Node count = 3
                (0, [2]),
                (2, []),
                (0, [1]),
            ],
            [ # Node count = 7
               (0, [1, 2]),
                (0, []),
                (0, [3]),
                (2, [5]),
                (5, []),
                (3, [4, 6]),
                (5, []),
            ],
            [ # Node count = 9
                (0, [4, 5]),
                (5, []),
                (4, []),
                (5, []),
                (0, [2, 7]),
                (0, [1, 3, 6, 8]),
                (5, []),
                (4, []),
                (5, []),
            ],
            [ # Node count = 4
                (0, [1]),
                (0, [2]),
                (1, [3]),
                (2, []),
            ],
            [ # Node count = 5
                (0, [4]),
                (2, []),
                (3, [1]),
                (4, [2]),
                (0, [3]),
            ],
    ]

    sampleCount = len(treeIndex2NodeIndex2NbrIndices)

    torch.manual_seed(4)
    hierarchyPropagator = HierarchyPropagator(
        encoded_node_vec_len,
        propagated_info_len,
        max_node_count,
        node_info_propagator_stack_depth,
        disable_batch_norm=True,
    )

    nodeInfosTensor = torch.rand(sampleCount, max_node_count, encoded_node_vec_len)

    nodeInfoPropagated2 = hierarchyPropagator.test_forward(
        treeIndex2NodeIndex2NbrIndices,
        nodeInfosTensor,
    )

    nodeInfoPropagated = hierarchyPropagator.forward(
        treeIndex2NodeIndex2NbrIndices,
        nodeInfosTensor,
        None,
    )
