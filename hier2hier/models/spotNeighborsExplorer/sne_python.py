from typing import Dict, Tuple, List

import torch

from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel, longTensor

class SpotNeighborsExplorerPy(ModuleBase):
    def __init__(self):
        super().__init__(None)

    def forward(self, graph, alreadySeenIn, activeNodesIn):
        """
            Inputs:
                graph: A tuple (nbrs, nbrCounts) representing the adjacency grpah of nodes.
                alreadySeenIn: A set of node.
                activeNodesIn: All neighbors of nodeSet, which don't fall back into nodeSet.
            Outputs:
                alreadySeenOut: alreadySeenSetIn \\union activeNodeSet
                activeNodeOut: nbrs(activeNodesIn) - alreadySeenSetOut
        """
        nbrs, nbrCounts = graph
        alreadySeenSet = set(alreadySeenIn.tolist())
        # print("Start pycode: {0}".format(alreadySeenSet))
        activeNodesSet = set()
        for activeNode in activeNodesIn.tolist():
            # print("For {0}".format(activeNode))
            nbrCount = int(nbrCounts[activeNode])
            for nbr in nbrs[activeNode, 0:nbrCount].tolist():
                if nbr not in alreadySeenSet:
                    # print("\tNot found {0}".format(nbr))
                    activeNodesSet.add(int(nbr))
                # else:
                    # print("\tFound {0}".format(nbr))
        # print("End pycode:")
        alreadySeenSet = alreadySeenSet.union(activeNodesSet)

        activeNodesOut = longTensor(sorted(activeNodesSet), device=activeNodesIn.device)
        alreadySeenOut = longTensor(sorted(alreadySeenSet), device=alreadySeenIn.device)

        return alreadySeenOut, activeNodesOut
