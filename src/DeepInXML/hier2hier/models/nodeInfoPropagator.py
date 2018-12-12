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
from .utils import invertPermutation, checkNans

class NodeInfoPropagator(nn.Module):
    def __init__(self,
            encoded_node_vec_len,
            propagated_info_len,
            max_node_count,
            node_info_propagator_stack_depth,
            device=None):
        super().__init__()
        self.encoded_node_vec_len = encoded_node_vec_len
        self.propagated_info_len = propagated_info_len
        self.max_node_count = max_node_count
        self.node_info_propagator_stack_depth = node_info_propagator_stack_depth
        self.device = device

        # Upgrade size of input.
        self.resizeInfoWidth = nn.Linear(self.encoded_node_vec_len, self.propagated_info_len)
        self.parentOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)
        self.neighborOp = nn.Linear(self.propagated_info_len, self.propagated_info_len)

        # Neighbor info gate.
        self.gruCell = torch.nn.GRUCell(propagated_info_len, propagated_info_len)

    def computeFanoutOrder(self, treeIndex2NodeIndex2NbrIndices):
        # For efficiency, we need to order nodes in the decreasing order of fanout.
        flatIndicesWithFanout = []
        flatIndex = 0
        for _, nodeIndex2NbrIndices in treeIndex2NodeIndex2NbrIndices.items():
            for _, (_, childIndices) in nodeIndex2NbrIndices.items():
                flatIndicesWithFanout.append((flatIndex, len(childIndices)))
                flatIndex += 1
        flatIndicesWithFanout.sort(key=lambda t: -t[1])

        # Compute the permutation to use.
        flatIndicesByDecreasingFanout = [ flatIndex for (flatIndex, _) in flatIndicesWithFanout ]
        decreasingFanouts = torch.tensor([ fanout for (_, fanout) in flatIndicesWithFanout], device=self.device)
        return flatIndicesByDecreasingFanout, decreasingFanouts

    def computeFlatReOrderedIndexMap(self,
            treeIndex2NodeIndex2NbrIndices,
            flatIndicesByDecreasingFanoutInverse):
        flatReOrderedIndexMap = {}
        flatIndex = 0
        for treeIndex, nodeIndex2NbrIndices in treeIndex2NodeIndex2NbrIndices.items():
            treeIndexOffset = flatIndex
            for nodeIndex, (parentIndex, childIndices) in nodeIndex2NbrIndices.items():
                flatIndexNewPos = flatIndicesByDecreasingFanoutInverse[flatIndex]
                parentIndexFlat = parentIndex + treeIndexOffset
                parentIndexNewPos = flatIndicesByDecreasingFanoutInverse[parentIndexFlat]
                childIndicesNewPos = []
                for childIndex in childIndices:
                    childIndexFlat = childIndex + treeIndexOffset
                    childIndexNewPos = flatIndicesByDecreasingFanoutInverse[childIndexFlat]
                    childIndicesNewPos.append(childIndexNewPos)
                flatReOrderedIndexMap[flatIndexNewPos] = (parentIndexNewPos, childIndicesNewPos)
                flatIndex += 1
        return flatReOrderedIndexMap

    def computeNeighborSelectors(self, flatReOrderedIndexMap):
        # Every node has a parent. Here, we find an order, which gets parent for each node.        
        selectorForParentInfos = []
        selectorForChildrenInfoList = []
        for flatIndexNewPos, (parentIndexNewPos, childIndicesNewPos) in flatReOrderedIndexMap.items():
            selectorForParentInfos.append(parentIndexNewPos)
            for i, childIndexNewPos in enumerate(childIndicesNewPos):
                if i == len(selectorForChildrenInfoList):
                    selectorForChildrenInfoList.append([])
                selectorForChildrenInfoList[i].append(childIndexNewPos)
        return selectorForParentInfos, selectorForChildrenInfoList

    def forward(self, treeIndex2NodeIndex2NbrIndices, nodeInfosTensor):
        # For efficiency, we need to re-arrange nodes in nodeInfosTensor in the increasing
        # order of fanout.
        (
            flatIndicesByDecreasingFanout,
            decreasingFanouts,
        ) = self.computeFanoutOrder(treeIndex2NodeIndex2NbrIndices)

        # flatIndicesByDecreasingFanout is a permutation of the nodes in nodeInfosTensor.
        # Compute inverse of this permutation.
        flatIndicesByDecreasingFanoutInverse = invertPermutation(flatIndicesByDecreasingFanout)

        # Translate all neighbor indices present in treeIndex2NodeIndex2NbrIndices, so that they
        # are all flat and re-arranged in the order of decreasing fanouts.
        flatReOrderedIndexMap = self.computeFlatReOrderedIndexMap(
            treeIndex2NodeIndex2NbrIndices,
            flatIndicesByDecreasingFanout)

        # Compute parent and children info selectors. These selectors are used in
        # propagation of node information to their neighbors.
        (
            selectorForParentInfos,
            selectorForChildrenInfoList,
        ) = self.computeNeighborSelectors(flatReOrderedIndexMap)

        sampleCount = len(nodeInfosTensor)
        checkNans(nodeInfosTensor)
        nodeInfoPropagated = self.resizeInfoWidth(nodeInfosTensor)
        checkNans(nodeInfoPropagated)

        # Get a flattened view of nodeInfoToPropagate. Flat view is easier to permute.
        nodeInfoPropagatedFlat = nodeInfoPropagated.view(sampleCount*self.max_node_count, self.propagated_info_len)

        # Permute nodeInfoToPropagateFlat in the order of decreasing fanout.
        nodeInfoPropagatedReOrdered = nodeInfoPropagatedFlat[flatIndicesByDecreasingFanout, ...]

        for i in range(self.node_info_propagator_stack_depth):
            # Prepare parent info for propagation into new nodeInfoPropagated.
            checkNans(nodeInfoPropagatedFlat)
            parentInfosToPropagateReOrdered = nodeInfoPropagatedFlat[selectorForParentInfos, ...]
            checkNans(parentInfosToPropagateReOrdered)

            # Compute children info to propagate to each node.
            childrenInfoToPropagateReOrdered = torch.tensor([], device=self.device)
            for selectorForChildrenInfo in selectorForChildrenInfoList:
                curChildrenInfoReOrdered = nodeInfoPropagatedReOrdered[selectorForChildrenInfo, ...]
                if not childrenInfoToPropagateReOrdered.shape[0]:
                    # First iteration of the loop.
                    childrenInfoToPropagateReOrdered = curChildrenInfoReOrdered
                else:
                    assert(curChildrenInfoReOrdered.shape[0] >= childrenInfoToPropagateReOrdered.shape[0])
                    # If the fanout increases in current iteration, pad neighbor infos by the deficit. 
                    if curChildrenInfoReOrdered.shape[0] > childrenInfoToPropagateReOrdered.shape[0]:
                        deficit = curChildrenInfoReOrdered.shape[0] - childrenInfoToPropagateReOrdered.shape[0]
                        childrenInfoToPropagateReOrdered = nn.ZeroPad2d(0, 0, 0, deficit)(childrenInfoToPropagateReOrdered)
                    childrenInfoToPropagateReOrdered = childrenInfoToPropagateReOrdered + curChildrenInfoReOrdered

            if childrenInfoToPropagateReOrdered.shape[0]:
                # Row-wise normalization of childrenInfoToPropagate by fanout.
                # Don't do it in-place.
                childrenInfoToPropagateReOrdered = childrenInfoToPropagateReOrdered / decreasingFanouts

                # There may still be some row deficit remaining because some nodes do not have children.
                finalDeficit = nodeInfoPropagatedFlat.shape[0] - childrenInfoToPropagateReOrdered.shape[0]
                childrenInfoToPropagateReOrdered = nn.ZeroPad2d(0, 0, 0, finalDeficit)(childrenInfoToPropagateReOrdered)
            else:
                # The case where no node has a child an all are in deficit.
                childrenInfoToPropagateReOrdered = torch.zeros(nodeInfoPropagatedFlat.shape, device=self.device)

            # Apply distinct linear operations on parent and children node infos before propagate.
            checkNans(parentInfosToPropagateReOrdered)
            parentInfosToPropagateReOrdered = self.parentOp(parentInfosToPropagateReOrdered)
            childrenInfoToPropagateReOrdered = self.neighborOp(childrenInfoToPropagateReOrdered)

            # Compute final neighbor information summary.
            neighborsNodeInfoSummary = parentInfosToPropagateReOrdered + childrenInfoToPropagateReOrdered

            # Propagate the new neighbor information using GRU cell to obtain updated node info.
            checkNans(nodeInfoPropagatedReOrdered)
            nodeInfoPropagatedReOrdered = self.gruCell(nodeInfoPropagatedReOrdered, neighborsNodeInfoSummary)
            checkNans(nodeInfoPropagatedReOrdered)

        # Invert re-ordering to obtain the flat newly propagated node info tensor.
        nodeInfoPropagated = nodeInfoPropagatedReOrdered[flatIndicesByDecreasingFanoutInverse, ...]

        # Undo flat.
        nodeInfoPropagated = nodeInfoPropagated.view(sampleCount, self.max_node_count, self.propagated_info_len)
        checkNans(nodeInfoPropagated)
        return nodeInfoPropagated

