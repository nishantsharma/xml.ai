from typing import Dict, Tuple, List

import torch

from hier2hier.models.moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel

useJit=False
TorchJitDecorator=torch.jit.script_method if useJit else (lambda x:x)
class AccumulateByValue(torch.jit.ScriptModule if useJit else ModuleBase):
    __constants__ = ['mode']
    def __init__(self, mode, schemaVersion):
        if mode=="max":
            self.mode = 0
        elif mode=="sum":
            self.mode = 1
        if useJit:
            super().__init__(None)
        else:
            super().__init__(None, schemaVersion)

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 0:
            pass
        else:
            super().singleStepSchema(schemaVersion)

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
            beamCount = int(valuesToAccumulate.shape[0])
            return torch.cat (
                [
                    self(
                        valuesToAccumulate[i,...],
                        vecIndicesToKey,
                        totalKeyCount,
                        False
                    ).unsqueeze(1)
                    for i in range(beamCount)
                ],
                1
            )
            

        numVectors = valuesToAccumulate.shape[0]
        remainingDims = valuesToAccumulate.shape[1:]

        indexBoundaries = torch.bincount(vecIndicesToKey, minlength=totalKeyCount)

        retval = []
        start = 0
        for key in range(totalKeyCount):
            end = start + int(indexBoundaries[key])
            toAppend = valuesToAccumulate[start:end]
            if self.mode == 0: # MAX
                if toAppend.shape[0] != 0:
                    toAppend = torch.max(toAppend)
                else:
                    toAppend = torch.zeros(toAppend.shape[1:], device=valuesToAccumulate.device)
            elif self.mode == 1: # SUM
                toAppend = torch.sum(toAppend, dim=0)
            toAppend = toAppend.unsqueeze(0)
            # toAppend = toAppend.view([1] + toAppend.shape)
            retval.append(toAppend)
            start=end

        return torch.cat(retval, 0)

class AccumulateSumByValue(AccumulateByValue):
    def __init__(self, schemaVersion):
        super().__init__("sum", schemaVersion)
        
class AccumulateMaxByValue(AccumulateByValue):
    def __init__(self, schemaVersion):
        super().__init__("max", schemaVersion)