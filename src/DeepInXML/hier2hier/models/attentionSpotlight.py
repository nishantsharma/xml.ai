from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys, copy
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from hier2hier.models.moduleBase import ModuleBase

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
                device=None):
        super().__init__(device)

        # Network for attention model between query and attention ready inputs.
        self.attentionCorrelation = nn.Bilinear(
            attnReadyVecLen,
            queryVecLen,
            headCount,
        )

    def reset_parameters(self, device):
        self.attentionCorrelation.reset_parameters()
        
    def initialSpotLight(self, attnReadyPosNbrhoodGraph):
        """
        Initially, look everywhere.
        """
        sli2gni = list(range(len(attnReadyPosNbrhoodGraph)))
        return torch.LongTensor(sli2gni, device=self.device)

    def forward(self,
                attnReadyPosNbrhoodGraph,
                attnReadyVecsByGni,
                prevSli2Gni,
                gni2Tdol,
                curQueryVec,
                treeCount,
                beamMode,
    ):
        """
        Inputs:
            attnReadyVecsByGni:
                Shape: graphSize X attnReadyVecLen
                Data: Attention ready vectors representing items that can be attended to.
            prevSli2Gni:
                Shape: len(prevSLI)
                Data: GNI index at each previous spotlight index position.
            gni2Tdol:
                Shape: len(GNI)
                Data: TDOL index of the XML tree corresponding to each GNI index.
            curQuery:
                if beamMode:
                    Shape: sampleCount X beamCount X queryVecLen
                Else:
                    Shape: sampleCount X queryVecLen
                Data: Vectors representing the query.
        Output:
            curSli2Gni
                Shape: len(curSLI)
                Data: GNI index at each current spotlight index position.
            attnReadyInfoCollapsedByTdol
                if beamMode:
                    Shape: sampleCount X beamCount X attnReadyVecLen
                Else:
                    Shape: sampleCount X attnReadyVecLen
                Data: Vectors representing the query.

        """
        sliDimension = 1 if beamMode else 0
        # As an optimization, we don't look at members already seen.
        alreadySeenGni = set(prevSli2Gni.tolist())

        prevSli2Tdol = gni2Tdol[prevSli2Gni]
        attentionFactorsByPrevSLI = self.computeAttentionFactors(
            attnReadyVecsByGni[prevSli2Gni],
            curQueryVec,
            prevSli2Tdol,
            beamMode=beamMode,
        )

        maxAttentionFactorByTDOL, _ = self.accumulatFactorsByTree(
            prevSli2Gni,
            attentionFactorsByPrevSLI,
            gni2Tdol,
            treeCount,
            torch.max,
            beamMode=beamMode,
        )

        def cullSmallFactors(discoveredGni, attentionFactors, maxAttentionFactorByTDOL):
            """
                Inputs
                    discoveredGni:
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
            discoveredGni2Tdol = gni2Tdol[discoveredGni]

            # Indexing below
            # (treeCount X beamCount) Indexing SliCount-Value-TDOL
            # Shape: sliCount X beamCount
            maxAttentionFactorToUse = maxAttentionFactorByTDOL[discoveredGni2Tdol]
            
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
                retainedIndicesBool = torch.sum(retainedIndicesBool, dim=0)

            retainedIndices = torch.LongTensor([
                i for i, _ in enumerate(discoveredGni)
                if retainedIndicesBool[i]
            ])
            return retainedIndices

        discoveredNewGniList = [ (prevSli2Gni, attentionFactorsByPrevSLI) ]
        # Next, figure out next set of members which should be added to the attention set.
        while True:
            # Next set of candidate members.
            discoveredNewGni = set()
            for curGni in discoveredNewGniList[-1][0]:
                discoveredNewGni = discoveredNewGni.union(
                    set(attnReadyPosNbrhoodGraph[int(curGni)]) - alreadySeenGni
                )

            # End loop if no new candidate found.
            if not(discoveredNewGni):
                break

            # For future, mark the candidate members as seen.
            alreadySeenGni = alreadySeenGni.union(discoveredNewGni)

            discoveredNewGni = torch.LongTensor(list(discoveredNewGni))

            discoveredNewGni2Tdol = gni2Tdol[discoveredNewGni]
            attentionFactorsForNewGni = self.computeAttentionFactors(
                attnReadyVecsByGni[discoveredNewGni],
                curQueryVec,
                discoveredNewGni2Tdol,
                beamMode=beamMode,
            )

            # Updae maxAttentionFactor for each tree.
            maxAttentionFactorByTDOL = torch.max(
                maxAttentionFactorByTDOL,
                self.accumulatFactorsByTree(
                    discoveredNewGni,
                    attentionFactorsForNewGni,
                    gni2Tdol,
                    treeCount,
                    torch.max,
                    beamMode=beamMode,
                )[0]
            )

            # Cull factors which are much smaller than max.
            retainedIndices = cullSmallFactors(
                discoveredNewGni,
                attentionFactorsForNewGni,
                maxAttentionFactorByTDOL,
            )
            discoveredNewGni = discoveredNewGni[retainedIndices]
            attentionFactorsForNewGni = torch.index_select(
                attentionFactorsForNewGni,
                sliDimension,
                retainedIndices,
            )

            # End loop if no new candidate found.
            if not(retainedIndices.shape):
                break

            # Add new members to nextEPoA.
            discoveredNewGniList.append((discoveredNewGni, attentionFactorsForNewGni))

        # Need to again cull small factors, as the maxAttentionFactorByTDOL has now
        # chnaged.
        curSli2Gni = []
        attentionFactorsByCurSli = []
        for discoveredNewGni, attentionFactorsForNewGni in discoveredNewGniList:
            retainedIndices = cullSmallFactors(
                discoveredNewGni,
                attentionFactorsForNewGni,
                maxAttentionFactorByTDOL,
            )
            curSli2Gni.append(discoveredNewGni[retainedIndices])
            attentionFactorsByCurSli.append(
                torch.index_select(
                    attentionFactorsForNewGni,
                    sliDimension, 
                    torch.LongTensor(retainedIndices)
                )
            )

        # Concatenate lists of tensors into a single tensor.
        curSli2Gni = torch.cat(curSli2Gni)
        attentionFactorsByCurSli = torch.cat(attentionFactorsByCurSli, dim=sliDimension)

        # To normalize each factor within a tree, we need to find sum of all factors, by TDOL.
        sumAttentionFactorByTDOL = self.accumulatFactorsByTree(
            curSli2Gni,
            attentionFactorsByCurSli,
            gni2Tdol,
            treeCount,
            torch.sum,
            beamMode=beamMode,
        )

        # Normalize attention factors.
        curSli2Tdol = gni2Tdol[curSli2Gni]
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
        attnReadyVecsBySli = attnReadyVecsByGni[curSli2Gni]
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
        attnReadyInfoCollapsedByTdol = self.accumulatFactorsByTree(
            curSli2Gni,
            normalizedAttentionFactorsByCurSli,
            gni2Tdol,
            treeCount,
            torch.sum,
            beamMode=beamMode,
        )

        if beamMode:
            # Switch dims to make shape beamCount X tdolCount.
            attnReadyInfoCollapsedByTdol = attnReadyInfoCollapsedByTdol.permute([1, 0, -1])

        return curSli2Gni, attnReadyInfoCollapsedByTdol

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
    def accumulatFactorsByTree(
            gniIndices,
            factorsForGniIndices,
            gni2Tdol,
            treeCount,
            accumulate,
            beamMode,
    ):
        """
            Inputs:
                gniIndices:
                    Shape: sliCount
                    Data: GNI index being used.
                factorsForGniIndices:
                    In beamMode:
                        Shape: beamCount X sliCount
                        Data: Factors
                    Else:
                        Shape: sliCount
                        Data: Factors
                gni2Tdol:
                    Shape: gniliCount
                    Data: TDOL index at each GNI position.
                treeCount: Number of trees.
                accumulate: Function to use for accumulation.
            Output:
                In beamMode:
                    Shape: tdolCount X beamCount
                    Data: Accumulated factors.
                Else:
                    Shape: tdolCount 
                    Data: Accumulated factors.
        """
        if beamMode:
            beamCount, sliCount = factorsForGniIndices.shape[0:2]
            remainingDims = list(factorsForGniIndices.shape[2:])
        else:
            sliCount = factorsForGniIndices.shape[0]
            remainingDims = list(factorsForGniIndices.shape[1:])
        device = factorsForGniIndices.device
        # Compute maximum scale factors within the same tree.
        gniIndicesList = list(range(len(gniIndices)))
        tdolIndicesTensor = gni2Tdol[gniIndices]
        gniIndices2OneHotTdol = torch.LongTensor(
            [
                tdolIndicesTensor.tolist(),
                gniIndicesList,
            ],
            device=device
        )
        if beamMode:
            # factorsForGniIndices permuted as sliCount X beamCount X remainingDims
            dimPerm = [1, 0] + [k+2 for k in range(len(remainingDims))]
            factorsForGniIndices = factorsForGniIndices.permute(dimPerm)

            # sparseFactorsForGniIndicesIntoTdol created as
            # tdolCount X sliCount X beamCount X remainingDims
            sparseFactorsForGniIndicesIntoTdol = torch.sparse.FloatTensor(
                gniIndices2OneHotTdol,
                factorsForGniIndices,
                torch.Size([treeCount, sliCount, beamCount] + remainingDims)
            )
        else:
            # sparseFactorsForGniIndicesIntoTdol created as
            # tdolCount X sliCount X remainingDims
            sparseFactorsForGniIndicesIntoTdol = torch.sparse.FloatTensor(
                gniIndices2OneHotTdol,
                factorsForGniIndices,
                torch.Size([treeCount, sliCount] + remainingDims)
            )

        try:
            retval = accumulate(sparseFactorsForGniIndicesIntoTdol, dim=1)
        except RuntimeError:
            retval = accumulate(sparseFactorsForGniIndicesIntoTdol.to_dense(), dim=1)

        return retval

