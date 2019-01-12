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

class NodeInfoPropagator(ModuleBase):
    def __init__(self,
            encoded_node_vec_len,
            propagated_info_len,
            max_node_count,
            node_info_propagator_stack_depth,
            disable_batch_norm,
            input_dropout_p=0,
            dropout_p=0,
            device=None):
        super().__init__(device)
        self.encoded_node_vec_len = encoded_node_vec_len
        self.propagated_info_len = propagated_info_len
        self.max_node_count = max_node_count
        self.node_info_propagator_stack_depth = node_info_propagator_stack_depth
        self.disable_batch_norm = disable_batch_norm

        # Upgrade size of input.
        self.resizeInfoWidth = nn.Linear(self.encoded_node_vec_len, self.propagated_info_len)

        self.input_dropout = nn.Dropout(p=input_dropout_p)

        # Batch norm on propagated info.
        if not self.disable_batch_norm:
            self.batchNormPropagatedInfo = nn.BatchNorm1d(num_features=propagated_info_len)

        # Ops to apply while propagating parent and neighbor info.
        self.parentOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)
        self.childrenOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)

        # Neighbor info gate.
        self.gruCell = torch.nn.GRUCell(propagated_info_len, propagated_info_len)

        # GRU output dropout.
        self.dropout = nn.Dropout(p=dropout_p)

    def reset_parameters(self):
        self.resizeInfoWidth.reset_parameters()
        if not self.disable_batch_norm:
            self.batchNormPropagatedInfo.reset_parameters()
        self.parentOp.reset_parameters()
        self.childrenOp.reset_parameters()
        self.gruCell.reset_parameters()

    @torch.no_grad()
    def test_forward(self, treeIndex2NodeIndex2NbrIndices, nodeInfosTensor):
        # Resize by width.
        nodeInfoPropagated = self.resizeInfoWidth(nodeInfosTensor)

        # Apply input dropout. Better to do after resize as input bits are non-uniform.
        nodeInfoPropagated = self.input_dropout(nodeInfoPropagated)

        if not self.disable_batch_norm:
            nodeInfoPropagated = self.batchNormPropagatedInfo(nodeInfoPropagated)

        for treeIndex, nodeIndex2NbrIndices in enumerate(treeIndex2NodeIndex2NbrIndices):
            for unOccupiedNodeIndex in range(len(nodeIndex2NbrIndices), self.max_node_count):
                nodeInfoPropagated[treeIndex, unOccupiedNodeIndex, ...] = 0

        for i in range(self.node_info_propagator_stack_depth):
            nodeInfoPropagatedNext = torch.zeros(nodeInfoPropagated.shape)
            for treeIndex, nodeIndex2NborIndices in enumerate(treeIndex2NodeIndex2NbrIndices):
                for nodeIndex, (parentIndex, childrenIndices) in enumerate(nodeIndex2NborIndices):
                    # Compute node info to propagate.
                    nodeInfoToPropagate = nodeInfoPropagated[treeIndex, nodeIndex]

                    # Compute parent info to propagate.
                    parentInfoToPropagate = nodeInfoPropagated[treeIndex, parentIndex]
                    parentInfoToPropagate = self.parentOp(parentInfoToPropagate)

                    # Compute child info to propagate.
                    childrenInfoToPropagate = torch.zeros(nodeInfoPropagated.shape[2:])
                    if childrenIndices:
                        for childIndex in childrenIndices:
                            childrenInfoToPropagate += nodeInfoPropagated[treeIndex, childIndex]
                        childrenInfoToPropagate /= len(childrenIndices)
                    childrenInfoToPropagate = self.childrenOp(childrenInfoToPropagate)

                    # Compute final neighbor information summary.
                    neighborsNodeInfoSummary = parentInfoToPropagate + childrenInfoToPropagate

                    # Propagate the new neighbor information using GRU cell to obtain updated node info.
                    nodeInfoPropagatedNext[treeIndex, nodeIndex] = self.gruCell(
                        nodeInfoToPropagate.view(1, self.propagated_info_len),
                        neighborsNodeInfoSummary.view(1, self.propagated_info_len))

            # Set the node info.
            nodeInfoPropagated = nodeInfoPropagatedNext


        return nodeInfoPropagated

    @methodProfiler
    def packInDecreasingNodeCountOrder(self,
        nodeInfoPropagated,
        treeIndex2NodeIndex2NbrIndices,
    ):
        # DNC = decreasing node count
        sampleCount = nodeInfoPropagated.shape[0]
        nodeCountsOrig = [
                len(nodeIndex2NbrIndices)
                for nodeIndex2NbrIndices in treeIndex2NodeIndex2NbrIndices
            ]

        # Permute trees in the order of node counts.
        dncTreeIndex2OrigTreeIndex = list(range(len(nodeCountsOrig)))
        dncTreeIndex2OrigTreeIndex.sort(key=lambda i:-nodeCountsOrig[i])
        origTreeIndex2DncTreeIndex = invertPermutation(dncTreeIndex2OrigTreeIndex)

        # Decreasing nod counts.
        decreasingNodeCounts = [nodeCountsOrig[i] for i in dncTreeIndex2OrigTreeIndex]

        # For indexing, we need LongTensors not lists.
        origTreeIndex2DncTreeIndex = torch.LongTensor(origTreeIndex2DncTreeIndex)
        dncTreeIndex2OrigTreeIndex = torch.LongTensor(dncTreeIndex2OrigTreeIndex)

        # Input trees to be in decreasing node count order.
        dncNodeInfoPropagated = torch.index_select(
            nodeInfoPropagated,
            0,
            dncTreeIndex2OrigTreeIndex)

        # Get a packed view of nodeInfoPropagated. Flat view is easier to permute.
        dncNodeInfoPropagatedPacked = rnn.pack_sequence([
            dncNodeInfoPropagated[dncTreeIndex, 0:decreasingNodeCounts[dncTreeIndex], ...]
            for dncTreeIndex in range(sampleCount)
        ]).data

        # Build origTreeNodeIndex2PackedNodeIndex.
        packedNodeIndex = 0
        origTreeNodeIndex2PackedNodeIndex = {}
        maxNodeIndexWithinTree = decreasingNodeCounts[0]
        for nodeIndexWithinTree in range(maxNodeIndexWithinTree):
            for dncTreeIndex in range(sampleCount):
                if nodeIndexWithinTree >= decreasingNodeCounts[dncTreeIndex]:
                    # This tree doesn't have any more nodes.
                    break
                origTreeIndex = dncTreeIndex2OrigTreeIndex[dncTreeIndex]
                origTreeNodeIndex2PackedNodeIndex[(int(origTreeIndex), nodeIndexWithinTree)] = packedNodeIndex
                packedNodeIndex += 1

        # Build dncTreeIndex2PackedNodeIndices.
        dncTreeIndex2PackedNodeIndices = OrderedDict(
            {
                dncTreeIndex:torch.LongTensor(
                    [
                        origTreeNodeIndex2PackedNodeIndex[(origTreeIndex, nodeIndexWithinTree)]
                        for nodeIndexWithinTree in range(len(treeIndex2NodeIndex2NbrIndices[origTreeIndex]))
                    ],
                    device=self.device
                )
                for origTreeIndex, dncTreeIndex in enumerate(origTreeIndex2DncTreeIndex.tolist())
            }
        )

        # Fixup neighbors.
        packedNodeIndices2PackedNbrs = OrderedDict({})
        for origTreeIndex, origNodeIndex2OrigNbrs in enumerate(treeIndex2NodeIndex2NbrIndices):
            for nodeIndexWithinTree, (parentIndexWithinTree, childIndicesWithinTree) in enumerate(origNodeIndex2OrigNbrs):
                packedNodeIndex = origTreeNodeIndex2PackedNodeIndex[
                    (origTreeIndex, nodeIndexWithinTree)
                ]

                packedParentIndex = origTreeNodeIndex2PackedNodeIndex[
                    (origTreeIndex, parentIndexWithinTree)
                ]

                packedChildrenIndices = [
                    origTreeNodeIndex2PackedNodeIndex[(origTreeIndex, childIndexWithinTree)]
                    for childIndexWithinTree in childIndicesWithinTree
                ]
                packedNodeIndices2PackedNbrs[packedNodeIndex] = (packedParentIndex, packedChildrenIndices)

        return (
                # DNC Permutation.
                dncTreeIndex2OrigTreeIndex,
                origTreeIndex2DncTreeIndex,

                # Packing Permutation
                origTreeNodeIndex2PackedNodeIndex,
                dncTreeIndex2PackedNodeIndices,
                packedNodeIndices2PackedNbrs,

                # Processed data.
                dncNodeInfoPropagatedPacked,
            )

    @methodProfiler
    def computePackedNodeIndicesinDFO(self, packedNodeIndices2PackedNbrs):
        """
        DFO -> Decreasing Fanout Order.
        """
        # For efficiency, we need to order nodes in the decreasing order of fanout.
        packedNodeIndicesWithFanout = [
            (packedNodeIndex, len(packedChildIndices))
            for packedNodeIndex, (_, packedChildIndices) in packedNodeIndices2PackedNbrs.items()
        ]
        packedNodeIndicesWithFanout.sort(key=lambda t: -t[1])

        # Compute the permutation to use.
        dfoIndex2PackedNodeIndex = [
            packedNodeIndex for (packedNodeIndex, _) in packedNodeIndicesWithFanout
        ]
        decreasingFanouts = torch.LongTensor([ fanout for (_, fanout) in packedNodeIndicesWithFanout if fanout], device=self.device)

        # Compute inverse of this dfoIndex2PackedNodeIndex.
        packedNodeIndex2DfoIndex = torch.LongTensor(
            invertPermutation(dfoIndex2PackedNodeIndex)
        )
        dfoIndex2PackedNodeIndex = torch.LongTensor(dfoIndex2PackedNodeIndex)

        return (
            dfoIndex2PackedNodeIndex,
            packedNodeIndex2DfoIndex,
            decreasingFanouts
        )

    @methodProfiler
    def computeNeighborSelectorsInDFO(
        self,
        packedNodeIndices2PackedNbrs,
        packedNodeIndex2DfoIndex,
    ):
        # Every node has a parent. Here, we find an order, which gets parent for each node.
        totalNodeCount = len(packedNodeIndices2PackedNbrs)
        dfoSelectorForParentInfos = [None for _ in range(totalNodeCount)]
        dfoSelectorForChildrenInfoList = []
        for packedNodeIndex, (packedParentIndex, packedChildIndices) in packedNodeIndices2PackedNbrs.items():
            dfoNodeIndex = int(packedNodeIndex2DfoIndex[packedNodeIndex])
            dfoParentIndex = int(packedNodeIndex2DfoIndex[packedParentIndex])
            dfoSelectorForParentInfos[dfoNodeIndex] = dfoParentIndex
            for i, packedChildIndex in enumerate(packedChildIndices):
                if i == len(dfoSelectorForChildrenInfoList):
                    dfoSelectorForChildrenInfoList.append({})
                dfoChildIndex = int(packedNodeIndex2DfoIndex[packedChildIndex])
                dfoSelectorForChildrenInfoList[i][dfoNodeIndex] = dfoChildIndex

        # Convert dict selectors to list selectors. Also reverse, because smallest children list first.
        dfoSelectorForChildrenInfoList = [
            [ dfoSelectorForChildrenInfo[k] for k in range(len(dfoSelectorForChildrenInfo)) ]
            for dfoSelectorForChildrenInfo in reversed(dfoSelectorForChildrenInfoList)
        ]

        # For torch indexing, we need LongTensor, not lists.
        for i in range(len(dfoSelectorForChildrenInfoList)):
            dfoSelectorForChildrenInfoList[i] = torch.LongTensor(dfoSelectorForChildrenInfoList[i])
        dfoSelectorForParentInfos = torch.LongTensor(dfoSelectorForParentInfos)

        return dfoSelectorForParentInfos, dfoSelectorForChildrenInfoList

    @methodProfiler
    def forward(self, treeIndex2NodeIndex2NbrIndices, nodeInfosTensor, tensorBoardHook):
        sampleCount = len(nodeInfosTensor)

        # Resize by width.
        nodeInfoPropagated = self.resizeInfoWidth(nodeInfosTensor)

        # Apply input dropout. Better to do after resize as input bits are non-uniform.
        nodeInfoPropagated = self.input_dropout(nodeInfoPropagated)

        # For packing, we need to arrange all trees in nodeInfoPropagated in
        # decreasing node count order(DNC order).
        (
            # DNC Permutation.
            dncTreeIndex2OrigTreeIndex,
            origTreeIndex2DncTreeIndex,

            # Packing Permutation
            origTreeNodeIndex2PackedNodeIndex,
            dncTreeIndex2PackedNodeIndices,
            packedNodeIndices2PackedNbrs,

            # Processed data.
            dncNodeInfoPropagatedPacked,
        ) = self.packInDecreasingNodeCountOrder(
            nodeInfoPropagated,
            treeIndex2NodeIndex2NbrIndices,
        )

        # Next, we need to re-arrange packed nodes in dncNodeInfoPropagatedPacked in the
        # increasing order of fanout.
        (
            dfoIndex2PackedNodeIndex,
            packedNodeIndex2DfoIndex,
            decreasingFanouts,
        ) = self.computePackedNodeIndicesinDFO(packedNodeIndices2PackedNbrs)

        # Permute dncNodeInfoPropagatedPacked in decreasing fanout order.
        nodeInfoPropagatedPackedDfoed = torch.index_select(
            dncNodeInfoPropagatedPacked,
            0,
            dfoIndex2PackedNodeIndex)

        # Compute parent and children info selectors. These selectors are used in
        # propagation of node information to their neighbors.
        (
            dfoSelectorForParentInfos,
            dfoSelectorForChildrenInfoList,
        ) = self.computeNeighborSelectorsInDFO(
            packedNodeIndices2PackedNbrs,
            packedNodeIndex2DfoIndex,
        )

        # decreasingFanoutsFactor is used to scale propagated children node info.
        decreasingFanoutsFactor = decreasingFanouts.float()
        decreasingFanoutsFactor = decreasingFanoutsFactor.view(decreasingFanouts.shape[0], 1)
        decreasingFanoutsFactor = decreasingFanoutsFactor.expand(-1, self.propagated_info_len)

        # Apply batch norm.
        if not self.disable_batch_norm:
            nodeInfoPropagatedPackedDfoed = self.batchNormPropagatedInfo(nodeInfoPropagatedPackedDfoed)

        # Run propagation loop.
        for i in range(self.node_info_propagator_stack_depth):
            # Prepare parent info for propagation into new nodeInfoPropagated.
            parentInfoPropagatedPackedDfoed = nodeInfoPropagatedPackedDfoed[dfoSelectorForParentInfos, ...]

            # Compute children info to propagate to each node.
            childrenInfoPropagatedPackedDfoed = torch.tensor([], device=self.device)
            for dfoSelectorForChildrenInfo in dfoSelectorForChildrenInfoList:
                curChildInfoPropagatedPackedDfoed = nodeInfoPropagatedPackedDfoed[dfoSelectorForChildrenInfo, ...]
                if not childrenInfoPropagatedPackedDfoed.shape[0]:
                    # First iteration of the loop.
                    childrenInfoPropagatedPackedDfoed = curChildInfoPropagatedPackedDfoed
                else:
                    assert(curChildInfoPropagatedPackedDfoed.shape[0] >= childrenInfoPropagatedPackedDfoed.shape[0])
                    # If the fanout increases in current iteration, pad neighbor infos by the deficit.
                    if curChildInfoPropagatedPackedDfoed.shape[0] > childrenInfoPropagatedPackedDfoed.shape[0]:
                        deficit = curChildInfoPropagatedPackedDfoed.shape[0] - childrenInfoPropagatedPackedDfoed.shape[0]
                        childrenInfoPropagatedPackedDfoed = nn.ZeroPad2d((0, 0, 0, deficit))(childrenInfoPropagatedPackedDfoed)
                    childrenInfoPropagatedPackedDfoed = childrenInfoPropagatedPackedDfoed + curChildInfoPropagatedPackedDfoed

            if childrenInfoPropagatedPackedDfoed.shape[0]:
                # Row-wise normalization of childrenInfoToPropagate by fanout.
                # Don't do it in-place.
                childrenInfoPropagatedPackedDfoed = childrenInfoPropagatedPackedDfoed / decreasingFanoutsFactor

                # There may still be some row deficit remaining because some nodes do not have children.
                finalDeficit = nodeInfoPropagatedPackedDfoed.shape[0] - childrenInfoPropagatedPackedDfoed.shape[0]
                childrenInfoPropagatedPackedDfoed = nn.ZeroPad2d((0, 0, 0, finalDeficit))(childrenInfoPropagatedPackedDfoed)
            else:
                # The case where no node has a child an all are in deficit.
                childrenInfoPropagatedPackedDfoed = torch.zeros(nodeInfoPropagatedFlat.shape, device=self.device)

            # Apply distinct linear operations on parent and children node infos before propagate.
            parentInfoPropagatedPackedDfoed = self.parentOp(parentInfoPropagatedPackedDfoed)
            childrenInfoPropagatedPackedDfoed = self.childrenOp(childrenInfoPropagatedPackedDfoed)

            # Compute final neighbor information summary.
            neighborsNodeInfoSummary = parentInfoPropagatedPackedDfoed + childrenInfoPropagatedPackedDfoed

            # Propagate the new neighbor information using GRU cell to obtain updated node info.
            nodeInfoPropagatedPackedDfoed = self.gruCell(nodeInfoPropagatedPackedDfoed, neighborsNodeInfoSummary)

            # Apply GRU output dropout.
            nodeInfoPropagatedPackedDfoed = self.dropout(nodeInfoPropagatedPackedDfoed)

        origTreeIndex2DfoNodeIndices = []
        for origTreeIndex in range(sampleCount):
            dncTreeIndex = int(origTreeIndex2DncTreeIndex[origTreeIndex])
            packedNodeIndices = dncTreeIndex2PackedNodeIndices[dncTreeIndex]
            dfoNodeIndices = packedNodeIndex2DfoIndex[packedNodeIndices]
            origTreeIndex2DfoNodeIndices.append(dfoNodeIndices)

        # Invert DFO and packing in one go.
        nodeInfoPropagated = rnn.pad_sequence([
                nodeInfoPropagatedPackedDfoed[dfoNodeIndices]
                for dfoNodeIndices in origTreeIndex2DfoNodeIndices
            ],
            batch_first = True,
        )

        # Sometimes, we need padding because max_node_count doesn't match.
        nodeCountDeficit = self.max_node_count - nodeInfoPropagated.shape[1]
        if  nodeCountDeficit != 0:
            assert(nodeCountDeficit>=0)
            nodeInfoPropagated = nn.ZeroPad2d((0, 0, 0, nodeCountDeficit))(nodeInfoPropagated)

        return nodeInfoPropagated


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
    nodeInfoPropagator = NodeInfoPropagator(
        encoded_node_vec_len,
        propagated_info_len,
        max_node_count,
        node_info_propagator_stack_depth,
        disable_batch_norm=True,
    )

    nodeInfosTensor = torch.rand(sampleCount, max_node_count, encoded_node_vec_len)

    nodeInfoPropagated2 = nodeInfoPropagator.test_forward(
        treeIndex2NodeIndex2NbrIndices,
        nodeInfosTensor,
    )

    nodeInfoPropagated = nodeInfoPropagator.forward(
        treeIndex2NodeIndex2NbrIndices,
        nodeInfosTensor,
        None,
    )
