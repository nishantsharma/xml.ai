"""
AttentionSpotlight is an implementation of attention mechanism over connectivity graph
of positions within an XML. All XML nodes, attributes and positions within any text field
are potential positions worth attending.
"""
from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys, copy, math
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel, checkNans, longTensor
from hier2hier.models.hier2hierBatch import splitByToi
from hier2hier.models.accumulateByValue import AccumulateByValue, AccumulateSumByValue, AccumulateMaxByValue
from hier2hier.models.spotNeighborsExplorer import SpotNeighborsExplorer

class AttentionSpotlight(ModuleBase):
    """
    Module used to identiy and track a subset of graph node indices for implementing attention.
    Orders
        GNI: Original order of ALL Graph Node Indices.
        SLI: Spot Light Indices. A subset of GNI indices which are under attention.
        prevSLI: The subset of GNI indices, which were under spotlight in the previous
                iteration.
        TDOL: XML Trees in Decreasing Output Lengths order(Applicable only during training.).
        1 HOT TDOL: Suppose we have a vector sli2tdol containing the TDOL index at each
                SLI position. That means sli2tdol = [tdol1, tdol2, ..., tdol_lastSLI ]
                Then we can "1hot encode" it as:
                sli2OneHotTdol = [ [0, .., 1, ..], oneHotTdol_k, ..., ]
        SPARSE 1 HOT TDOL: A (1 HOT TDOL) encoded tensor, which is also torch sparse.
        PrevSLI_x_SOHTDOL, SLI_x_SOHTDOL: SLI X SPARSE 1 HOT TDOL.
    """
    def __init__(self,
                spotlightThreshold=None,
                headCount=1,
                device=None,
                checkGraph=True,
    ):
        super().__init__(device)
        self.checkGraph = checkGraph
        self.spotlightThreshold = spotlightThreshold

        # Network for attention model between query and attention ready inputs.
        self.accumulateSumByValue = AccumulateSumByValue()
        self.accumulateMaxByValue = AccumulateMaxByValue()
        self.spotNeighborsExplorer = SpotNeighborsExplorer(device=device)

        self.batchNormWeights = nn.BatchNorm1d(num_features=headCount)

    def reset_parameters(self, device):
        self.batchNormWeights.reset_parameters()

    @methodProfiler
    def forward(self,
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                posEncodedVecsByGndtol,
                prevSli2Gndtol,
                gndtol2Tdol,
                curQueryVec,
                treeCount,
                tensorBoardHook,
                beamMode=False,
                spotlightThreshold=None,
                dataDebugHook=None,
                debugAttention=False,
    ):
        """
        Inputs:
            attnReadyVecsByGndtol:
                Shape: graphSize X attnReadyVecLen
                Data: Vectors used to compute attention factors.
            posEncodedVecsByGndtol:
                Shape: graphSize X posENcodedVecLen
                Data: Attention ready vectors representing items that can be attended to.
            prevSli2Gndtol:
                Shape: len(prevSLI)
                Data: GNI index at each previous spotlight index position.
            gndtol2Tdol:
                Shape: len(GNI)
                Data: TDOL index of the XML tree corresponding to each GNI index.
            curQuery:
                if beamMode:
                    Shape: sampleCount X beamCount X queryVecLen
                Else:
                    Shape: sampleCount X queryVecLen
                Data: Vectors representing the query.
        Output:
            curSli2Gndtol
                Shape: len(curSLI)
                Data: GNI index at each current spotlight index position.
            attnReadyInfoCollapsedByTdol
                if beamMode:
                    Shape: sampleCount X beamCount X attnReadyVecLen
                Else:
                    Shape: sampleCount X attnReadyVecLen
                Data: Vectors representing the query.

        """
        if dataDebugHook is None:
            def dataDebugHook(*argv, **kargv):
                pass

        if spotlightThreshold is None:
            spotlightThreshold = self.spotlightThreshold

        prevSli2Tdol = gndtol2Tdol[prevSli2Gndtol]
        preExpAattentionFactorsByPrevSLI, attentionFactorsByPrevSLI = self.computeAttentionFactors(
            attnReadyVecsByGndtol[prevSli2Gndtol],
            curQueryVec,
            prevSli2Tdol,
            beamMode=beamMode,
        )
        tensorBoardHook.add_histogram("preExpAattentionFactorsByPrevSLI", preExpAattentionFactorsByPrevSLI)
        dataDebugHook(attentionFactorsByPrevSLI, "tdol", indexList=prevSli2Tdol)

        sliDimension = 1 if beamMode else 0

        if self.checkGraph:
            maxAttentionFactorByTDOL = self.accumulateMaxByValue(
                attentionFactorsByPrevSLI,
                gndtol2Tdol[prevSli2Gndtol],
                treeCount,
                beamMode=beamMode,
            )

            dataDebugHook(maxAttentionFactorByTDOL, "tdol")
 
            # As an optimization, we don't look at members already seen.
            alreadySeenSli2Gndtol = prevSli2Gndtol
            discoveredNewGndtol = prevSli2Gndtol

            discoveredNewGndtolList = [ (prevSli2Gndtol, attentionFactorsByPrevSLI) ]
            # Next, figure out next set of members which should be added to the attention set.
            while True:
                # Find next set of candidate members by exploring neighbors.
                (
                    alreadySeenSli2Gndtol,
                    discoveredNewGndtol,
                ) = self.spotNeighborsExplorer(
                    posNbrhoodGraphByGndtol,
                    alreadySeenSli2Gndtol,
                    discoveredNewGndtol,
                )

                # End loop if no new candidate found.
                if not(discoveredNewGndtol.shape[0]):
                    break

                discoveredNewGndtol2Tdol = gndtol2Tdol[discoveredNewGndtol]
                _, attentionFactorsForNewGndtol = self.computeAttentionFactors(
                    attnReadyVecsByGndtol[discoveredNewGndtol],
                    curQueryVec,
                    discoveredNewGndtol2Tdol,
                    beamMode=beamMode,
                )

                # Update maxAttentionFactor for each tree.
                maxAttentionFactorByTDOL = torch.max(
                    maxAttentionFactorByTDOL,
                    self.accumulateMaxByValue.wrapper(
                        attentionFactorsForNewGndtol,
                        gndtol2Tdol[discoveredNewGndtol],
                        treeCount,
                        beamMode=beamMode,
                    )
                )

                # Cull factors which are much smaller than max.
                retainedIndices = self.cullSmallFactors(
                    gndtol2Tdol,
                    beamMode,
                    discoveredNewGndtol,
                    attentionFactorsForNewGndtol,
                    maxAttentionFactorByTDOL,
                    spotlightThreshold,
                )
                if retainedIndices is False:
                    # No index retained.
                    break

                if retainedIndices is not True:
                    # Only some indices are retained.
                    discoveredNewGndtol = discoveredNewGndtol[retainedIndices]
                    attentionFactorsForNewGndtol = torch.index_select(
                        attentionFactorsForNewGndtol,
                        sliDimension,
                        retainedIndices,
                    )

                # Add new members to nextEPoA.
                discoveredNewGndtolList.append((discoveredNewGndtol, attentionFactorsForNewGndtol))

            # Need to again cull small factors, as the maxAttentionFactorByTDOL has now
            # chnaged.
            curSli2Gndtol = []
            attentionFactorsByCurSli = []
            for discoveredNewGndtol, attentionFactorsForNewGndtol in discoveredNewGndtolList:
                retainedIndices = self.cullSmallFactors(
                    gndtol2Tdol,
                    beamMode,
                    discoveredNewGndtol,
                    attentionFactorsForNewGndtol,
                    maxAttentionFactorByTDOL,
                    spotlightThreshold,
                )
                if retainedIndices is False:
                    # No index retained.
                    continue

                if retainedIndices is not True:
                    # Only some indices are retained.
                    discoveredNewGndtol = discoveredNewGndtol[retainedIndices]
                    attentionFactorsForNewGndtol = torch.index_select(
                        attentionFactorsForNewGndtol,
                        sliDimension,
                        retainedIndices,
                    )

                # Insert into new list.
                curSli2Gndtol.append(discoveredNewGndtol)
                attentionFactorsByCurSli.append(attentionFactorsForNewGndtol)

            # Concatenate lists of tensors into a single tensor.
            curSli2Gndtol = torch.cat(curSli2Gndtol)
            attentionFactorsByCurSli = torch.cat(attentionFactorsByCurSli, dim=sliDimension)

            dataDebugHook(attentionFactorsByCurSli, "gndtol", indexList=curSli2Gndtol)

            # Sort all results.
            curSli2Gndtol, sortingPermutation = torch.sort(curSli2Gndtol)
            attentionFactorsByCurSli = torch.index_select(
                attentionFactorsByCurSli,
                sliDimension,
                sortingPermutation,
            )
        else:
            curSli2Gndtol = prevSli2Gndtol
            attentionFactorsByCurSli = attentionFactorsByPrevSLI

        tensorBoardHook.add_histogram("attentionFactorsByCurSli", attentionFactorsByCurSli)
        # To normalize each factor within a tree, we need to find sum of all factors, by TDOL.
        sumAttentionFactorByTDOL = self.accumulateSumByValue.wrapper(
            attentionFactorsByCurSli,
            gndtol2Tdol[curSli2Gndtol],
            treeCount,
            beamMode=beamMode,
        )

        # Normalize attention factors.
        curSli2Tdol = gndtol2Tdol[curSli2Gndtol]
        attentionFactorsDivisor = sumAttentionFactorByTDOL[curSli2Tdol]
        if beamMode:
            # In beam mode:
            #     sumAttentionFactorByTDOL[curSli2Tdol]
            #     We are indexing (tdolCount X beamCount) by sliCount-value-tdol
            # Resulting shape = sliCount X beamCount
            # Permute it to get beamCount X sliCount 
            attentionFactorsDivisor = attentionFactorsDivisor.permute(1, 0)

        # Normalize by sum along each tree.
        normalizedAttentionFactorsByCurSli = attentionFactorsByCurSli/attentionFactorsDivisor
        tensorBoardHook.add_histogram("normalizedAttentionFactorsByCurSli", normalizedAttentionFactorsByCurSli)

        # Extend to add the vector width dimension.
        normalizedAttentionFactorsByCurSli = normalizedAttentionFactorsByCurSli.unsqueeze(-1)

        # Get vecs by SLI.
        posEncodedVecsBySli = posEncodedVecsByGndtol[curSli2Gndtol]
        if beamMode:
            # Add a beam dimension to attnReadyVecsBySli.
            posEncodedVecsBySli = posEncodedVecsBySli.unsqueeze(0)

        if debugAttention:
           normalizedFactorsByGndtol = torch.zeros(posNbrhoodGraphByGndtol[1].shape, device=self.device, requires_grad=False)
           normalizedFactorsByGndtol[curSli2Gndtol] = normalizedAttentionFactorsByCurSli.squeeze(-1)
        else:
           normalizedFactorsByGndtol = None

        # Use factors to obtain linear combination of cur SLI subset attnReadyVecs.
        normalizedAttentionVectorsByCurSli = (
            posEncodedVecsBySli
            *
            normalizedAttentionFactorsByCurSli
        )

        # Accumulate along SLI dimension.
        posEncodedVecsCollapsedByTdol = self.accumulateSumByValue.wrapper(
            normalizedAttentionVectorsByCurSli,
            gndtol2Tdol[curSli2Gndtol],
            treeCount,
            beamMode=beamMode,
        )

        checkNans(posEncodedVecsCollapsedByTdol)

        return curSli2Gndtol, posEncodedVecsCollapsedByTdol, normalizedFactorsByGndtol

    @methodProfiler
    def computeAttentionFactors(self,
            attnReadyVecs,
            curQueryVec,
            sli2Tdol,
            beamMode,
    ):
        """
            Inputs:
                attnReadyVecs
                    Shape: sliCount X attnReadyVecLen.
                    Data: Attention ready vectors of each
                curQueryVec:
                    In beam mode:
                        Shape: sampleCount X beamCount X queryVecLen
                        Data: Query vector for which we want to find the best match.
                    else:
                        Shape: sampleCount X queryVecLen
                        Data: Query vector for which we want to find the best match.
            Output:
                In beam mode:
                    Shape: beamCount X sliCount
                    Data: (queryVec Dot attnReadyVec)
                else:
                    Shape: sliCount
                    Data: (queryVec Dot attnReadyVec)
        """
        # Re-map curQueryVec as sliCount X queryVecLen
        curQueryVec = curQueryVec[sli2Tdol]

        if beamMode:
            # Reshape attnReadyVecs as beamCount X sliCount X attnReadyVecLen
            beamCount = curQueryVec.shape[1]
            attnReadyVecs = attnReadyVecs.unsqueeze(0)
            attnReadyVecs = attnReadyVecs.expand(beamCount, -1, -1).unsqueeze(2)

            # Reshape curQueryVec as beamCount X sliCount X queryVecLen
            curQueryVec = curQueryVec.permute(1, 0, 2).unsqueeze(3)
        else:
            attnReadyVecs = attnReadyVecs.unsqueeze(1)
            curQueryVec = curQueryVec.unsqueeze(2)

        # Take batch dot product using matmul.
        preExpAttnFactors = torch.matmul(attnReadyVecs, curQueryVec).squeeze(-1)

        # Clamp between range -50 to +50.
        preExpAttnFactors = torch.clamp(preExpAttnFactors, max=20)

        # Apply batchNormWeights.
        restoreShape = preExpAttnFactors.shape
        preExpAttnFactors = self.batchNormWeights(preExpAttnFactors.view(-1, 1)).view(restoreShape)

        expAttnFactors = torch.exp(preExpAttnFactors)
        assert(expAttnFactors.shape[-1] == 1)
        expAttnFactors = expAttnFactors.view(beamCount, -1) if beamMode else expAttnFactors.view(-1)

        return preExpAttnFactors, expAttnFactors

    @staticmethod
    def cullSmallFactors(
        gndtol2Tdol,
        beamMode,
        discoveredGndtol,
        attentionFactors,
        maxAttentionFactorByTDOL,
        spotlightThreshold,
    ):
        """
            Inputs
                discoveredGndtol:
                    Shape: sliCount
                attentionFactors:
                    if beamMode:
                        Shape: beamCount X sliCount
                    else:
                        Shape: sliCount
                maxAttentionFactorByTDOL
                    if beamMode:
                        Shape: treeCount X beamCount
                    else:
                        Shape: treeCount
            Outputs:
        """
        # Get TDOL of each SLI.
        #   Shape: sliCount
        #   Value: tree index of the SLI.
        discoveredGndtol2Tdol = gndtol2Tdol[discoveredGndtol]

        # Indexing below
        # if beamMode:
        #     Shape: sliCount X beamCount
        # else:
        #     Shape: sliCount
        maxAttentionFactorToUse = maxAttentionFactorByTDOL[discoveredGndtol2Tdol]
        
        if beamMode:
            # Shape: beamCount X sliCount
            # Permute last two dimensions to make it ready for comparison.
            maxAttentionFactorToUse = maxAttentionFactorToUse.permute(1, 0)

        # Purpose of comparison is to cull small(1/1000) factors.
        maxAttentionFactorToUse *= spotlightThreshold

        # Compare.
        # Shape: beamCount X SliCount.
        retainedIndicesBool = (attentionFactors > maxAttentionFactorToUse)

        if beamMode:
            # Collapse along beamCount dimension.
            # Retain if any beam is suggesting retention.
            retainedIndicesBool = (torch.sum(retainedIndicesBool, dim=0) != 0)

        retainedCount = torch.sum(retainedIndicesBool)
        if retainedCount == 0:
            return False
        elif retainedCount == len(retainedIndicesBool):
            return True

        retainedIndices = longTensor([
            i for i, _ in enumerate(discoveredGndtol)
            if retainedIndicesBool[i]
        ], device=retainedIndicesBool.device)
        return retainedIndices

def attentionSpotlightUnitTest():
    treeIndex2NodeIndex2NbrIndices = [
            [ # Node count = 9. Linear.
                [1], #0
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [],
            ],
            [ # Node count = 7. Reverse linear.
                [],
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
            ],
            [ # Node count = 6. Broken.
                [1, 2],
                [0],
                [0],
                [5],
                [5],
                [3, 4],
            ],
            [ # Node count = 5. Star network
                [1,2,3,4],
                [0],
                [0],
                [0],
                [0],
            ],
    ]
    treeCount = len(treeIndex2NodeIndex2NbrIndices)

    posNbrhoodGraphByGndtol = []
    treeOffset = 0
    attnReadyVecsByGndtol = []
    gndtol2Tdol = []
    for treeIndex, nodeIndex2NbrIndices in enumerate(treeIndex2NodeIndex2NbrIndices):
        for nodeIndex, nbrIndices in enumerate(nodeIndex2NbrIndices):
            curNbrList = [childIndex+treeOffset for childIndex in nbrIndices]
            posNbrhoodGraphByGndtol.append(curNbrList)
            theta = 2*math.pi*nodeIndex/len(nodeIndex2NbrIndices)
            attnReadyVecsByGndtol.append([math.cos(theta), math.sin(theta)])
            gndtol2Tdol.append(treeIndex)
        treeOffset += len(nodeIndex2NbrIndices)
    gndtol2Tdol = torch.LongTensor(gndtol2Tdol)

    # Convert lists to tensors.
    lengths = torch.LongTensor([len(n) for n in posNbrhoodGraphByGndtol])
    posNbrhoodGraphByGndtol = (
        torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(n)
                for n in posNbrhoodGraphByGndtol
            ]
        ),
        lengths
    )
    attnReadyVecsByGndtol = torch.tensor(attnReadyVecsByGndtol)

    # Build queryVec.
    queryVec=torch.tensor([[1.0, 0.0] for _ in range(treeCount)])

    prevSli2Gndtol = [0]
    for tree in treeIndex2NodeIndex2NbrIndices[0:-1]:
        prevSli2Gndtol.append(prevSli2Gndtol[-1] + len(tree))
    prevSli2Gndtol = torch.LongTensor(prevSli2Gndtol)
    
    # Get results
    attentionSpotlight = AttentionSpotlight(
                queryVec.shape[1],
    )

    # Get results
    results = attentionSpotlight(
        posNbrhoodGraphByGndtol,
        attnReadyVecsByGndtol,
        posEncodedVecsByGndtol,
        prevSli2Gndtol,
        gndtol2Tdol,
        queryVec,
        len(treeIndex2NodeIndex2NbrIndices),
        False,
        spotlightThreshold=0.5,
    )
