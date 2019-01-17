from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys, copy
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel
from hier2hier.models.accumulateByValue import AccumulateByValue, AccumulateSumByValue, AccumulateMaxByValue
from hier2hier.models.spotNeighborsExplorer import SpotNeighborsExplorer

class AttentionSpotlight(ModuleBase):
    """
    Module used to identiy and track a subset of graph node indices for implementing attention.
    Orders
        GNI: Original order of ALL Graph Node Indices.
        SLI: Spot Light Indices. A subset of GNI indices which are under attention.
        prevSLI: The subset of GNI indices, which were under potlight in the previous
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
                attnReadyVecLen,
                queryVecLen,
                headCount=1,
                device=None,
                checkGraph=True):
        super().__init__(device)
        self.checkGraph = checkGraph

        # Network for attention model between query and attention ready inputs.
        self.attentionCorrelation = nn.Bilinear(
            attnReadyVecLen,
            queryVecLen,
            headCount,
        )
        self.accumulateSumByValue = AccumulateSumByValue()
        self.accumulateMaxByValue = AccumulateMaxByValue()
        self.spotNeighborsExplorer = SpotNeighborsExplorer()

    def reset_parameters(self, device):
        self.attentionCorrelation.reset_parameters()

    @methodProfiler
    def forward(self,
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                prevSli2Gndtol,
                gndtol2Tdol,
                curQueryVec,
                treeCount,
                beamMode,
    ):
        """
        Inputs:
            attnReadyVecsByGndtol:
                Shape: graphSize X attnReadyVecLen
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
        prevSli2Tdol = gndtol2Tdol[prevSli2Gndtol]
        attentionFactorsByPrevSLI = self.computeAttentionFactors(
            attnReadyVecsByGndtol[prevSli2Gndtol],
            curQueryVec,
            prevSli2Tdol,
            beamMode=beamMode,
        )

        sliDimension = 1 if beamMode else 0

        if self.checkGraph:
            maxAttentionFactorByTDOL = self.accumulateMaxByValue.wrapper(
                attentionFactorsByPrevSLI,
                gndtol2Tdol[prevSli2Gndtol],
                treeCount,
                beamMode=beamMode,
            )

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
                ) = self.spotNeighborsExplorer.wrapper(
                    posNbrhoodGraphByGndtol,
                    alreadySeenSli2Gndtol,
                    discoveredNewGndtol,
                )

                # End loop if no new candidate found.
                if not(discoveredNewGndtol.shape[0]):
                    break

                discoveredNewGndtol2Tdol = gndtol2Tdol[discoveredNewGndtol]
                attentionFactorsForNewGndtol = self.computeAttentionFactors(
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

        # Extend to add the vector width dimension.
        normalizedAttentionFactorsByCurSli = normalizedAttentionFactorsByCurSli.view(
            list(normalizedAttentionFactorsByCurSli.shape) + [1]
        )

        # Get vecs by SLI.
        attnReadyVecsBySli = attnReadyVecsByGndtol[curSli2Gndtol]
        if beamMode:
            # Add a beam dimension to attnReadyVecsBySli.
            attnReadyVecsBySli = attnReadyVecsBySli.view([1] + list(attnReadyVecsBySli.shape))

        # Use factors to obtain linear combination of cur SLI subset attnReadyVecs.
        normalizedAttentionFactorsByCurSli = (
            attnReadyVecsBySli
            *
            normalizedAttentionFactorsByCurSli
        )

        # Accumulate along SLI dimension.
        attnReadyInfoCollapsedByTdol = self.accumulateSumByValue.wrapper(
            normalizedAttentionFactorsByCurSli,
            gndtol2Tdol[curSli2Gndtol],
            treeCount,
            beamMode=beamMode,
        )

        if beamMode:
            # Switch dims to make shape beamCount X tdolCount.
            attnReadyInfoCollapsedByTdol = attnReadyInfoCollapsedByTdol.permute([1, 0, -1])

        return curSli2Gndtol, attnReadyInfoCollapsedByTdol

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
            attnReadyVecs = attnReadyVecs.view([1] + list(attnReadyVecs.shape))
            attnReadyVecs = attnReadyVecs.expand(beamCount, -1, -1)

            # Reshape curQueryVec as beamCount X sliCount X queryVecLen
            curQueryVec = curQueryVec.permute(1, 0, 2)

        preExpAttnFactors = self.attentionCorrelation(attnReadyVecs.contiguous(), curQueryVec.contiguous())
        expAttnFactors = torch.exp(preExpAttnFactors)
        expAttnFactors = expAttnFactors.view(beamCount, -1) if beamMode else expAttnFactors.view(-1)

        return expAttnFactors

    @staticmethod
    def cullSmallFactors(
        gndtol2Tdol,
        beamMode,
        discoveredGndtol,
        attentionFactors,
        maxAttentionFactorByTDOL
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
        # (treeCount X beamCount) Indexing SliCount-Value-TDOL
        # Shape: sliCount X beamCount
        maxAttentionFactorToUse = maxAttentionFactorByTDOL[discoveredGndtol2Tdol]
        
        if beamMode:
            # Permute last two dimensions to make it ready for comparison.
            # Shape: beamCount X sliCount
            maxAttentionFactorToUse = maxAttentionFactorToUse.permute(1, 0)

        # Purpose of comparison is to cull small(1/1000) factors.
        maxAttentionFactorToUse /= 1000

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

        retainedIndices = torch.LongTensor([
            i for i, _ in enumerate(discoveredGndtol)
            if retainedIndicesBool[i]
        ])
        return retainedIndices

