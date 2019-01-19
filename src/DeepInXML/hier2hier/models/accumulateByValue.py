from typing import Dict, Tuple, List

import torch

from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel

test=True
TorchJitDecorator=(lambda x:x) if test else torch.jit.script_method
class AccumulateByValue(ModuleBase if test else torch.jit.ScriptModule):
    __constants__ = ['mode']
    def __init__(self, mode):
        if mode=="max":
            self.mode = 0
        elif mode=="sum":
            self.mode = 1
        super().__init__(None)

    @methodProfiler
    def wrapper(self, *argc, **kargv):
        return self(*argc, **kargv)

    @TorchJitDecorator
    def forward(self,
                valuesToAccumulate,
                vecIndicesToKey,
                totalKeyCount,
                beamMode,
    ):
        # type: (Tensor, Tensor, int, bool) -> Tensor
        """
            Inputs:
                valuesToAccumulate:
                    In beamMode:
                        Shape: beamCount X numVectors
                        Data: Factors
                    Else:
                        Shape: numVectors
                        Data: Factors
                vecIndicesToKey:
                    Shape: len(valuesToAccumulate)
                    Data: valuesToAccumulate vector index to accumultion key.
                totalKeyCount: Total Number of keys.
            Output:
                In beamMode:
                    Shape: totalKeyCount X beamCount
                    Data: Accumulated factors.
                Else:
                    Shape: totalKeyCount 
                    Data: Accumulated factors.
        """
        if beamMode:
            beamCount, numVectors = valuesToAccumulate.shape[0:2]
            remainingDims = valuesToAccumulate.shape[2:]
            valuesToAccumulate = valuesToAccumulate.permute([1, 0] + remainingDims)
        else:
            numVectors = valuesToAccumulate.shape[0]
            remainingDims = valuesToAccumulate.shape[1:]

        indexBoundaries = torch.bincount(vecIndicesToKey, minlength=totalKeyCount)

        retval = []
        start = 0
        for key in range(totalKeyCount):
            end = start + indexBoundaries[key]
            toAppend = valuesToAccumulate[start:end]
            if self.mode == 0: # MAX
                if toAppend.shape[0] != 0:
                    toAppend = torch.max(toAppend)
                else:
                    toAppend = torch.zeros(toAppend.shape[1:])
            elif self.mode == 1: # SUM
                toAppend = torch.sum(toAppend, dim=0)
            toAppend = toAppend.view([1] + list(toAppend.shape))
            retval.append(toAppend)
            start=end

        return torch.cat(retval, 0)

class AccumulateSumByValue(AccumulateByValue):
    def __init__(self):
        super().__init__("sum")
        
class AccumulateMaxByValue(AccumulateByValue):
    def __init__(self):
        super().__init__("max")

@methodProfiler
def accumulateFactorsByTree(
        gniIndices,
        tdolIndices,
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
    appendProfilingLabel("." + accumulate.__name__ + str(len(factorsForGniIndices.shape)))
    with blockProfiler("ListProcessing"):
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
    with blockProfiler("SparseTensorBuilding"):
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

    with blockProfiler("Accumulation"):
        try:
            retval = accumulate(sparseFactorsForGniIndicesIntoTdol, dim=1)
        except RuntimeError:
            retval = accumulate(sparseFactorsForGniIndicesIntoTdol.to_dense(), dim=1)

    return retval

