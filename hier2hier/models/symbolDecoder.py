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

        assert(schemaVersion >= 1)
        self.nonPointingShaper = nn.Linear(output_decoder_state_width, len(tgtVocabs.all))
        self.softMax = nn.Softmax(dim=-1)
        self.accumulateByValue = AccumulateSumByValue(schemaVersion)
        self.symbolSrcWeightsNetwork = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(output_decoder_state_width, int(output_decoder_state_width/2))),
            ("Selu1", nn.SELU()),
            ("Linear2", nn.Linear(int(output_decoder_state_width/2), int(output_decoder_state_width/3))),
            ("Selu2", nn.SELU()),
            ("Linear3", nn.Linear(int(output_decoder_state_width/3), 1)),
            ("BN", nn.BatchNorm1d(1)),
            ("Sigmoid", nn.Sigmoid()),
        ]))

    def reset_parameters(self, device):
        self.nonPointingShaper.reset_parameters()
        for component in self.symbolSrcWeightsNetwork:
            if list(component.parameters()):
                component.reset_parameters()
        self.softMax.reset_parameters()

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is -1:
            # Curently no migration is possible.
            pass
        else:
            super().singleStepSchema(schemaVersion)

    def forward(self,
            curGruOutputByTdol,
            srcSymbolsBySli,
            sli2Tdol,
            attnFactorsBySli,
            sampleCount,
            tensorBoardHook,
            beamMode=False,
            ):
        """
            Inputs:
                curGruOutput
                    Data: Output of the GRU RNN.
                    Shape: 
                        if BeamMode:
                            numSamples X beamCount X output_decoder_state_width
                        else
                            numSamples X output_decoder_state_width
                srcSymbolsBySli
                    Data: The output symbol that would be generated, if the input symbol at
                        corresponing graph position is just copied.
                    Shape: graphSize X outputVocabSize
                attnFactorsBySli
                    Data: Attention factors for each position in the graph.
                    Shape: graphSize
            Output:
                tgtSymbolProbsByTdol:
                    Data:
                        Softmax upon linear Combination of the following two.
                            Hypothesis One: Output symbol cn be generated from original src XML.
                                Collapse(srcSymbolsBySli into (graphSize X TDOL),
                                    using attnFactorsBySli)
                            Hypothesis Two: Output symbol must come primarily from gruOutput.
                                as self.shape(curGruOutputByTdol)
                    Shape: 
                        if beamMode:
                            outputVocabSize X beamCount X sampleCount
                        else:
                            outputVocabSize X sampleCount
        """
        nonPointerSymbolsByTdol = self.nonPointingShaper(curGruOutputByTdol)
        if self.useSrcPtr:
            # Combine with symbolDecoder and shaper to obtain the expected symbol.
            srcEquivalentSymbolsByTdol = self.accumulateByValue(
                attnFactorsBySli * srcSymbolsBySli,
                sli2Tdol,
                sampleCount,
                beamMode,
            )

            # Use curGruState to compute weight of direct copy versus derived value.
            if self.training and curGruOutputByTdol.shape[0] == 1:
                # Batch norm doesn't work on single sized batch.
                # Temporarily change to eval.
                self.symbolSrcWeightsNetwork.eval()
                symbolSrcWeights = self.symbolSrcWeightsNetwork(curGruOutputByTdol)
                self.symbolSrcWeightsNetwork.train()
            else:
                if beamMode:
                    origShape = curGruOutputByTdol.shape
                    curGruOutputByTdol = curGruOutputByTdol.view(origShape[0] * origShape[1], origShape[2])
                symbolSrcWeights = self.symbolSrcWeightsNetwork(curGruOutputByTdol)

                if beamMode:
                    curGruOutputByTdol = curGruOutputByTdol.view(origShape)
                    symbolSrcWeights = symbolSrcWeights.view(origShape[0], origShape[1], -1)

            tensorBoardHook.add_histogram("symbolSrcWeights", symbolSrcWeights)

            # Compute combined activation.
            combinedSymbolActivationByTdol = srcEquivalentSymbolsByTdol * symbolSrcWeights
            combinedSymbolActivationByTdol += nonPointerSymbolsByTdol * (1 - symbolSrcWeights)
        else:
            combinedSymbolActivationByTdol = nonPointerSymbolsByTdol

        return self.softMax(combinedSymbolActivationByTdol)


