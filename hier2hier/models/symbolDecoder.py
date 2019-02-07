from collections import OrderedDict 
import torch.nn as nn

from .moduleBase import ModuleBase
from .accumulateByValue import AccumulateSumByValue

class SymbolDecoder(ModuleBase):
    def __init__(self,
            schemaVersion,
            useSrcPtr,
            tgtVocabs,
            output_decoder_state_width,
            device,
    ):
        super().__init__(schemaVersion, device)
        self.useSrcPtr = useSrcPtr
        self.tgtVocabs = tgtVocabs

        if schemaVersion >= 1:
            self.shaper = nn.Linear(output_decoder_state_width, len(tgtVocabs.all))

            self.softMax = nn.Softmax(dim=-1)

            self.accumulateByValue = AccumulateSumByValue(schemaVersion)

            self.symbolSrcWeightsNetwork = nn.Sequential(OrderedDict([
                ("Linear1", nn.Linear(output_decoder_state_width, int(output_decoder_state_width/2))),
                ("Selu1", nn.SELU()),
                ("Linear2", nn.Linear(int(output_decoder_state_width/2), int(output_decoder_state_width/3))),
                ("Selu2", nn.SELU()),
                ("Linear3", nn.Linear(int(output_decoder_state_width/3), 1)),
                ("Sigmoid", nn.Sigmoid()),
            ]))
        else:
            raise NotImplementedError("SymbolDecoder in v1.")

    def reset_parameters(self, device):
        self.shaper.reset_parameters()
        self.softMax.reset_parameters()

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is -1:
            # Curently no migration is possible.
            pass
        else:
            super().singleStepSchema(schemaVersion)

    def forward(self,
            curGruOutputByTdol,
            srcSymbolsByGndtol,
            gndtol2Tdol,
            attnFactorsByGndtol,
            sampleCount,
            beamMode=False,
            ):
        """
            Inputs:
                curGruOutput
                    Data: Output of the GRU RNN.
                    Shape: numSamples X output_decoder_state_width
                srcSymbolsByGndtol
                    Data: The output symbol that would be generated, if the input symbol at
                        corresponing graph position is just copied.
                    Shape: graphSize X outputVocabSize
                attnFactorsByGndtol
                    Data: Attention factors for each position in the graph.
                    Shape: graphSize
            Output:
                tgtSymbolProbsByTdol:
                    Data:
                        Softmax upon linear Combination of the following two.
                            Hypothesis One: Output symbol cn be generated from original src XML.
                                Collapse(srcSymbolsByGndtol into (graphSize X TDOL),
                                    using attnFactorsByGndtol)
                            Hypothesis Two: Output symbol must come primarily from gruOutput.
                                as self.shape(curGruOutputByTdol)
                    Shape: outputVocabSize X sampleCount
        """
        nonPointerSymbolsByTdol = self.shaper(curGruOutputByTdol)
        if self.useSrcPtr:
            # Combine with symbolDecoder and shaper to obtain the expected symbol. 
            srcEquivalentSymbolsByTdol = self.accumulateByValue(
                attnFactorsByGndtol.unsqueeze(1) * srcSymbolsByGndtol,
                gndtol2Tdol,
                sampleCount,
                beamMode,
            )

            # Use curGruState to compute weight of direct copy versus derived value.
            symbolSrcWeights = self.symbolSrcWeightsNetwork(curGruOutputByTdol)

            # Compute combined activation.
            combinedSymbolActivationByTdol = srcEquivalentSymbolsByTdol * symbolSrcWeights
            combinedSymbolActivationByTdol += nonPointerSymbolsByTdol * (1 - symbolSrcWeights)
        else:
            combinedSymbolActivationByTdol = nonPointerSymbolsByTdol

        return self.softMax(combinedSymbolActivationByTdol)


