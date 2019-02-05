"""
    Torch Script implementaiton of Spot Neighbors Explorer.

    In SpotNeighborsExplorer, we explore neighbors of the attention spotlight for increasing
    attention factors. If the attentoin factors are above a certain threshold, we will include
    them in the next iteration of attention spotlight. 
"""
from typing import Dict, Tuple, List

import torch

from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel

class SpotNeighborsExplorerTS(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 0:
            pass
        else:
            super().singleStepSchema(schemaVersion)

    @torch.jit.script_method
    def forward(self, graph, alreadySeenSet, activeNodeSet):
        # type: (Tuple[Tensor, Tensor], Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """
            Inputs:
                graph: A tuple (nbrs, nbrCounts) representing the adjacency grpah of nodes.
                alreadySeenSetIn: A set of node.
                activeNodeSetIn: All neighbors of nodeSet, which don't fall back into nodeSet.
            Outputs:
                alreadySeenSetOut: alreadySeenSetIn \\union activeNodeSet
                activeNodeSetOut: nbrs(activeNodeSet) - alreadySeenSetOut
        """
        #with blockProfiler("nbrBuilder"):
        nbrs, nbrCounts = graph
        neighborSet = []
        for nodeIndex in range(activeNodeSet.shape[0]):
            node = activeNodeSet[nodeIndex]
            neighborSet.append(nbrs[node][0:nbrCounts[node]])
        neighborSet, _ = torch.sort(torch.cat(neighborSet))

        #with blockProfiler("sortNbrs"):
        alreadySeenSet, _ = torch.sort(torch.cat([alreadySeenSet, activeNodeSet]))

        #with blockProfiler("removeNbrDuplicates"):
        # Remove duplicates from alreadySeenSetOut.
        i, k = 0, 0
        n=len(alreadySeenSet)
        while i < n:
            j = i+1
            while j < n and int(alreadySeenSet[i]) == int(alreadySeenSet[j]):
                j += 1
            alreadySeenSet[k] = alreadySeenSet[i]
            k += 1
            i = j
        alreadySeenSet = alreadySeenSet[0:k]

        #with blockProfiler("removeActiveDuplicates"):
        # Remove duplicates from neighborSet, while excluding alreadySeenSetOut.
        i, j, k = 0, 0, 0
        activeNodeSet = neighborSet
        n = len(activeNodeSet)
        m = len(alreadySeenSet)
        while i < n:
            l = i+1
            while l < n and int(activeNodeSet[i]) == int(activeNodeSet[l]):
                l += 1
            while k < m and int(alreadySeenSet[k]) < int(activeNodeSet[i]):
                k += 1
            if k == m or int(alreadySeenSet[k]) != int(activeNodeSet[i]):
                activeNodeSet[j] = activeNodeSet[i]
                j += 1
            i = l
        activeNodeSet=activeNodeSet[0:j]

        return alreadySeenSet, activeNodeSet
