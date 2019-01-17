from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys, copy
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .moduleBase import ModuleBase
from .attentionSpotlight import AttentionSpotlight
from .beamSearch import BeamSearch
from hier2hier.util import (onehotencode, checkNans, blockProfiler,
                            methodProfiler, lastCallProfile)

import torch.nn.functional as F

class OutputDecoder(ModuleBase):
    def __init__(self,
            output_vocab,
            propagated_info_len,
            output_decoder_state_width,
            output_decoder_stack_depth,
            max_output_len,
            sos_id,
            eos_id,
            teacher_forcing_ratio=0,
            input_dropout_p=0,
            dropout_p=0,
            device=None,
            runtests=False):
        super().__init__(device)
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.output_decoder_stack_depth = output_decoder_stack_depth
        self.max_output_len = max_output_len
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = output_vocab.stoi["<pad>"]
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gruInputLen = propagated_info_len + len(output_vocab)

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.gruCell = nn.GRU(
            self.gruInputLen,
            self.output_decoder_state_width,
            self.output_decoder_stack_depth,
            batch_first=True,
            dropout=dropout_p)
        self.symbolsTensor = nn.Parameter(torch.eye(len(output_vocab), device=self.device), requires_grad=False)

        # Module for decoding attention and moving spotlight.
        self.attentionSpotlight = AttentionSpotlight(
            propagated_info_len,
            output_decoder_state_width)

        # Network for symbol decoding.
        self.symbolPreDecoder = nn.Linear(output_decoder_state_width, len(output_vocab))
        self.symbolDecoder = nn.Softmax(dim=-1)
        self.output_vocab = output_vocab
        self.runtests = runtests

        # Parameters required for loop initialization.
        self.initGruOutput = nn.Parameter(torch.zeros(self.output_decoder_state_width, ))
        self.buildInitGruState = nn.Linear(
            self.propagated_info_len,
            self.output_decoder_stack_depth  * self.output_decoder_state_width)

    def reset_parameters(self, device):
        self.gruCell.reset_parameters()
        self.attentionSpotlight.reset_parameters(device)
        nn.init.normal_(self.initGruOutput)
        self.buildInitGruState.reset_parameters()

    @torch.no_grad()
    def test_forward(self,
            nodeInfoPropagatedTensor,
            targetOutputsByTdol=None,
            targetOutputLengthsByTdol=None,
            teacherForcedSelections=None):
        # Dropout parts of data.
        nodeInfoPropagatedTensor = self.input_dropout(nodeInfoPropagatedTensor)

        sampleCount = len(nodeInfoPropagatedTensor)
        self.training = targetOutputsByTdol is not None

        outputSymbolsTensor = []
        outputSymbols = []
        for treeIndex in range(sampleCount):
            # Start symbol.
            curSymbol = self.sos_id
            curSymbolTensor = torch.tensor(
                [onehotencode(len(self.output_vocab), curSymbol)],
                device=self.device)

            # Initial gruOutput to emulate.
            curGruOutput = self.initGruOutput.view(1, 1, -1)

            # Initialize curGruState to None. It is updated within the loop, again.
            curGruState = None

            # Check if teaching is being forced.
            if not self.training or teacherForcedSelections is None:
                teacherForced = False
            else:
                teacherForced = teacherForcedSelections[treeIndex]

            treeOutputSymbolsTensor = [ curSymbolTensor ]
            treeOutputSymbols = [ curSymbol ]

            # Get the propagated node info of the current tree.
            nodeInfosPropagated = nodeInfoPropagatedTensor[treeIndex]

            maxSymbolIndex = -1
            for symbolIndex in range(1, self.max_output_len):

                use("Use attentionSpotlight to get new propagatedInfoToAttend.") 
                # Compute next attention.
                curAttention = self.decodeAttention(
                    nodeInfosPropagated.view(1, self.max_node_count, -1),
                    curGruOutput)

                # Compute node info summary to use in computation.
                attnReadyInfoCollapsedByTdol = torch.mm(
                        curAttention,
                        nodeInfosPropagated,
                    ).view(1, -1)

                # Build GRU Input.
                curGruInput = torch.cat([attnReadyInfoCollapsedByTdol, curSymbolTensor], -1)
                curGruInput = curGruInput.view(1, 1, self.gruInputLen)

                # For first iteration, we also need to build GRU State.
                if curGruState is None:
                    curGruState = self.buildInitGruState(attnReadyInfoCollapsedByTdol)
                    curGruState = curGruState.view(
                        self.output_decoder_stack_depth,
                        1,
                        self.output_decoder_state_width,
                    )

                # RUn GRU logic.
                curGruOutput, curGruState = self.gruCell(curGruInput, curGruState)

                # Compute next symbol tensor.
                generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
                generatedSymbolTensor = generatedSymbolTensor.view(1, len(self.output_vocab))
                treeOutputSymbolsTensor.append(generatedSymbolTensor)

                # Compute next symbol.
                if not self.training:
                    generatedSymbol = int(generatedSymbolTensor.topk(1)[1])
                    treeOutputSymbols.append(generatedSymbol)

                # Loop completion checks.
                if ((self.training and symbolIndex == targetOutputLengthsByTdol[treeIndex]-1)
                        or (not self.training and generatedSymbol == self.eos_id)):
                    paddingNeeded = self.max_output_len - len(treeOutputSymbolsTensor)
                    padTensor = torch.tensor([
                        onehotencode(
                            len(self.output_vocab),
                            self.pad_id
                        )])
                    treeOutputSymbolsTensor += [padTensor for _ in range(paddingNeeded)]

                    maxSymbolIndex = max(maxSymbolIndex, symbolIndex)
                    break


                ###################################################################
                ##### LOOP ITERATION COMPLETE. Now preping for next iteration #####
                ###################################################################
                if teacherForced:
                    curSymbolTensor = onehotencode(
                        len(self.output_vocab),
                        int(targetOutputsByTdol[treeIndex, symbolIndex]))
                    curSymbolTensor = torch.tensor([curSymbolTensor])
                else:
                    curSymbolTensor = generatedSymbolTensore

            treeOutputSymbolsTensor = torch.cat(treeOutputSymbolsTensor)
            outputSymbolsTensor.append(treeOutputSymbolsTensor)
            outputSymbols.append(treeOutputSymbols)

        outputSymbolsTensor = torch.cat(outputSymbolsTensor).view(
            sampleCount,
            int(self.max_output_len),
            -1
            )

        return outputSymbolsTensor

    def decodeSymbol(self, curGruOutput):
        symbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
        symbolTensor = symbolTensor.view(len(curGruOutput), len(self.output_vocab))
        symbols = [int(symbol) for symbol in symbolTensor.topk(1)[1].view(len(curGruOutput))]
        return symbolTensor, symbols

    def decodeAttention(self, nodeInfoPropagatedTensor, curGruOutput):
        sampleCount = len(curGruOutput)
        curGruOutput = curGruOutput.view(sampleCount, self.output_decoder_state_width)
        curGruOutput = curGruOutput.repeat(1, self.max_node_count)
        curGruOutput = curGruOutput.view(sampleCount, self.max_node_count, self.output_decoder_state_width)
        attentionInput = torch.cat([nodeInfoPropagatedTensor, curGruOutput], -1)
        return self.attentionDecoder(self.attentionPreDecoder(attentionInput).view(sampleCount, self.max_node_count))

    @methodProfiler
    def computeDimSqueezePoints(self, outputLimitsInOrder):
        """
        Compute the positions in output symbol computation, where we exclude another batch of trees from
        further consideration. We do that because the excluded output trees have their symbol computation already
        completed and need no more computation. Only used during training, when target output lengths are available.

        Input:
            outputLimitsInOrder: Length of target outputs in decreasing order.

        Output:
            dimSqueezePoints: List of tuples (outputIndexLimit, sampleIndexLimit)
                [(oil1, sil1), (oil2, sil2), (oil3, sil3), (oil4, sil4), ]
                For output indices [0, 1, ..., oil1-1] we use sampleIndexLimit as sil1.
                For output indices [oil1, oil1+1, ..., oil2-1] we use sampleIndexLimit as sil2.
                For output indices [oil2, oil2+1, ..., oil3-1] we use sampleIndexLimit as sil3.
                .
                .
                For output indices [oil2ndLast, oil2ndLast+1, ..., oilLast-1] we use sampleIndexLimit as silLast.

        """
        dimSqueezePoints = []
        outputLimitsInOrder = [ int(outputLimit) for outputLimit in outputLimitsInOrder ]
        sampleCount = len(outputLimitsInOrder)
        curOutputLimit = outputLimitsInOrder[-1]

        dimSqueezePoints.append((curOutputLimit, sampleCount))

        for sampleLimit, outputLimit in enumerate(outputLimitsInOrder[::-1]):
            if outputLimit == curOutputLimit:
                continue

            curOutputLimit = outputLimit
            dimSqueezePoints.append((curOutputLimit, sampleCount - sampleLimit))

        return dimSqueezePoints

    @methodProfiler
    def forward(self,
            sampleCount,
            posNbrhoodGraphByGndtol,
            initSpotlight,
            attnReadyVecsByGndtol,
            targetOutputsByTdol,
            targetOutputLengthsByTdol,
            gndtol2Tdol,
            tdol2Toi,
            tensorBoardHook,
            attentionSpotLight=None,
            collectOutput=None,
            beam_count=None,
            max_output_len=None,
        ):
        if max_output_len is None:
            max_output_len=max(self.max_output_len, max_output_len)

        # Dropout parts of data for robustness.
        attnReadyVecsByGndtol = self.input_dropout(attnReadyVecsByGndtol)

        if collectOutput is None:
            collectOutput = not(self.training)

        if not self.training and beam_count is not None:
            return self.beamSearch(
                sampleCount,
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                gndtol2Tdol,
                tdol2Toi,
                beam_count)

        with blockProfiler("ProcessPreLoop"):
            if self.training:
                # Get inverse of the TL order.
                dimSqueezePoints = self.computeDimSqueezePoints(targetOutputLengthsByTdol)
            else:
                dimSqueezePoints = [(max_output_len, sampleCount)]

            # Initial spotlight indices.
            sli2Gndtol = initSpotlight

            # Build first symbol input of the loop.
            curSymbol = self.sos_id
            curSymbolTensor = torch.tensor(
                [
                    onehotencode(len(self.output_vocab), curSymbol)
                    for _ in range(sampleCount)
                ],
                device=self.device
            )

            # Build first GruOutput of the loop.
            curGruOutput = torch.cat([self.initGruOutput.view(1, -1) for _ in range(sampleCount)])

            # GruState is initialized within the loop.
            curGruState = None

            # It is 1 at sampleIndex when teaching is being forced, else 0.
            if self.training:
                teacherForcedSelections = [
                    True if random.random() < self.teacher_forcing_ratio else False
                    for n in range(sampleCount)
                ]
            else:
                teacherForcedSelections = None

            # Init output vars.
            outputSymbolTensors = [ curSymbolTensor ]
            if collectOutput:
                outputSymbols = [ [ self.sos_id for _ in range(sampleCount) ] ]
            else:
                outputSymbols = None

        with blockProfiler("Loop"):
            for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
                if curGruOutput.shape[0] != sampleIndexLimit:
                    # Clip loop variables, restricting sample indices to sampleIndexLimit.
                    curSymbolTensor = curSymbolTensor.narrow(0, 0, sampleIndexLimit)
                    curGruState = curGruState.narrow(1, 0, sampleIndexLimit)
                    curGruOutput = curGruOutput.narrow(0, 0, sampleIndexLimit)

                    # Shorten sli2gndtol by dropping all trees
                    # with TDOl index >= sampleIndexLimit.
                    remainingSliIndices = (gndtol2Tdol[sli2Gndtol] < sampleIndexLimit)
                    remainingSliIndices = torch.LongTensor([
                        index
                        for index, enabled in enumerate(remainingSliIndices.tolist())
                        if enabled
                    ])
                    sli2Gndtol = sli2Gndtol[remainingSliIndices]
                    if targetOutputsByTdol is not None:
                        targetOutputsByTdol = targetOutputsByTdol.narrow(0, 0, sampleIndexLimit)

                # Logic for applying
                if self.training:
                    teacherForcingApplicator = torch.tensor(
                        [
                            n + teacherForcedSelections[n] * sampleIndexLimit
                            for n in range(sampleIndexLimit)
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )

                while True:
                    # Current symbol index can be determined by looking at output lists.
                    symbolIndex = len(outputSymbolTensors)
                    if symbolIndex == outputIndexLimit:
                        # We are crossing output index limit. So, this loop is over.
                        break

                    # Compute next attention.
                    with blockProfiler("ATTENTION-DECODE"):
                        (
                            sli2Gndtol,
                            attnReadyInfoCollapsedByTdol
                        ) = self.attentionSpotlight(
                            posNbrhoodGraphByGndtol,
                            attnReadyVecsByGndtol,
                            sli2Gndtol,
                            gndtol2Tdol,
                            curGruOutput,
                            sampleIndexLimit,
                            beamMode=False,
                        )

                    with blockProfiler("CATVIEW"):
                        curGruInput = torch.cat([attnReadyInfoCollapsedByTdol, curSymbolTensor], -1)
                        curGruInput = curGruInput.view(sampleIndexLimit, 1, self.gruInputLen)

                    if curGruState is None:
                        with blockProfiler("BuildInitGruState"):
                            curGruState = self.buildInitGruState(attnReadyInfoCollapsedByTdol)
                            curGruState = curGruState.view(
                                sampleCount,
                                self.output_decoder_stack_depth,
                                self.output_decoder_state_width,
                            ).permute(1, 0, 2)

                    with blockProfiler("GRUCELL1"):
                        curGruStateContiguous = curGruState.contiguous()

                    # Cycle RNN state.
                    with blockProfiler("GRUCELL2"):
                        tensorBoardHook.add_histogram("OutputDecoderGru.Input", curGruInput)
                        tensorBoardHook.add_histogram("OutputDecoderGru.State", curGruStateContiguous)
                        curGruOutput, curGruState = self.gruCell(curGruInput, curGruStateContiguous)
                        curGruOutput = curGruOutput.view(sampleIndexLimit, self.output_decoder_state_width)

                    # Compute next symbol.
                    with blockProfiler("SYMBOL-DECODE"):
                        generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
                        generatedSymbolTensor = generatedSymbolTensor.view(sampleIndexLimit, len(self.output_vocab))
                        outputSymbolTensors.append(generatedSymbolTensor)

                    # Compute next symbol list.
                    if collectOutput:
                        with blockProfiler("TOPK"):
                            generatedSymbol = [int(symbol) for symbol in generatedSymbolTensor.topk(1)[1].view(sampleIndexLimit)]
                            outputSymbols.append(generatedSymbol)


                    ###########################################################################
                    ##### CURRENT LOOP ITERATION COMPLETE. Now preping for next iteration #####
                    ###########################################################################

                    if self.training and symbolIndex == targetOutputsByTdol.shape[1]-1:
                        # We are crossing output index limit. So, this loop is over.
                        break

                    if self.training:
                        with blockProfiler("TEACHER-FORCING"):
                            targetSymbolsTensor = self.symbolsTensor[
                                targetOutputsByTdol[..., symbolIndex]
                            ]
                            concatedSymbolTensors = torch.cat([
                                generatedSymbolTensor, # Regular input.
                                targetSymbolsTensor, # Teaching forced.
                            ])

                            curSymbolTensor = torch.index_select(concatedSymbolTensors, 0, teacherForcingApplicator)
                    else:
                        curSymbolTensor = generatedSymbolTensor

        with blockProfiler("ProcessPostLoop"):
            symbolColumnsCount = len(outputSymbolTensors)
            if collectOutput:
                outputSymbolsTransposed = [[] for _ in range(sampleCount)]
                for curSymbolColumn in outputSymbols:
                    for j, curSymbol in enumerate(curSymbolColumn):
                        outputSymbolsTransposed[tdol2Toi[j]].append(curSymbol)
                outputSymbols = outputSymbolsTransposed

            # Pad symbol tensor columns to include pad symbols.
            padTensor = torch.tensor([onehotencode(len(self.output_vocab), self.pad_id)], device=self.device)
            for i, curSymbolTensorColumn in enumerate(outputSymbolTensors):
                paddingNeeded = sampleCount-len(curSymbolTensorColumn)
                if paddingNeeded:
                    outputSymbolTensors[i] = torch.cat(
                        [curSymbolTensorColumn]
                        + [ padTensor for _ in range(paddingNeeded)]
                    )

            outputSymbolTensors = torch.cat(
                [
                    outputSymbolTensor.view([sampleCount, 1, len(self.output_vocab)])
                    for outputSymbolTensor in outputSymbolTensors
                ],
                1 # Concat along second dimension.
            )

            # During training, we permute the columns
            if self.training:
                outputSymbolTensors = torch.index_select(outputSymbolTensors, 0, tdol2Toi)

            if self.runtests:
                padTensorColumn = torch.cat([padTensor.view(1, 1, -1) for _ in range(sampleCount)])
                decoderTestTensors = torch.cat(
                    [
                        outputSymbolTensors
                    ]
                    +
                    [
                        padTensorColumn
                        for _ in range(max_output_len - symbolColumnsCount)
                    ],
                    -2
                )
            else:
                decoderTestTensors = None

            checkNans(outputSymbolTensors)
            return outputSymbolTensors, outputSymbols, teacherForcedSelections, decoderTestTensors

    @methodProfiler
    def beamSearch(self,
                sampleCount,
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                gndtol2Tdol,
                tdoil2Toi,
                beam_count,
    ):
        # Initial spotlight indices.
        sli2Gndtol = self.attentionSpotlight.initialSpotLight(posNbrhoodGraphByGndtol)

        bigSampleCount = sampleCount * beam_count
        # Build first GruOutput to kickstart the loop.
        initGruOutput = torch.cat([self.initGruOutput.view(1, -1) for _ in range(sampleCount)])

        initGruState = None

        def decoderModel(prevBeamStatesTuple, prevBeamSymbols):
            """
                Method representing the output decoder training to the beamSearch method,
                Inputs:
                    prevBeamStatesTuple:
                        Tuple (prevGruOutput, prevGruState)
                        prevGruOutput:
                            Shape: SampleCount * beamCount * GruOutputVecLen
                        prevGruState:
                            Shape: SampleCount * beamCount * GruStateVecLen
                    prevBeamSymbolsIn:
                        Shape: sampleCount * beamCount
                        Data: ID of the beam symbol output in the last iteration.
                Outputs:
                    curBeamStatesOut:
                        Tuple (curGruOutput, curGruState)
                    curBeamSymbolProbsOut:
                        Shape: sampleCount * beamCount * vocabLen
                        Data: Values to be treated as log probabilities.
            """
            sampleCount, beamCount = prevBeamSymbols.shape
            vocabLen = self.symbolsTensor.shape[0]
            bigSampleCount = sampleCount * beamCount
            (prevGruOutput, prevGruState) = prevBeamStatesTuple
            nonlocal sli2Gndtol

            (
                sli2Gndtol,
                attnReadyInfoCollapsedByTdol,
            ) = self.attentionSpotlight(
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                sli2Gndtol,
                gndtol2Tdol,
                prevGruOutput,
                sampleCount,
                beamMode=True,
            )

            # sampleCount X beamCount X * -> bigSampleCount X *
            attnReadyInfoCollapsedByTdol = attnReadyInfoCollapsedByTdol.view(bigSampleCount, -1)
            prevBeamSymbols = prevBeamSymbols.view(bigSampleCount)

            # Concat to get combined output.
            prevBeamSymbols = self.symbolsTensor[prevBeamSymbols]
            curGruInput = torch.cat([attnReadyInfoCollapsedByTdol, prevBeamSymbols], -1)
            curGruInput = curGruInput.view(bigSampleCount, 1, self.gruInputLen)

            if prevGruState is None:
                prevGruState = self.buildInitGruState(attnReadyInfoCollapsedByTdol)

            prevGruState = prevGruState.view(
                bigSampleCount,
                self.output_decoder_stack_depth,
                self.output_decoder_state_width,
            )

            # Make num_layers dimension as dim 0.
            prevGruState = prevGruState.permute(1, 0, 2)

            # Need contiguous state for RNN.
            prevGruStateContiguous = prevGruState.contiguous()

            # Cycle RNN state.
            curGruOutput, curGruState = self.gruCell(curGruInput, prevGruStateContiguous)

            # Undo making num_layers dimension as dim 0.
            curGruState = curGruState.permute(1, 0, 2)

            # Compute next symbol.
            generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
            generatedSymbolTensor = generatedSymbolTensor.view(
                sampleCount, beamCount, vocabLen
            )

            # Split the dim0 of length into bigSampleCount two dims.
            curGruOutput = curGruOutput.view(
                [sampleCount, beamCount]
                + list(curGruOutput.shape[2:])
            )

            # Split the dim0 of length into bigSampleCount two dims.
            curGruState = curGruState.view(
                [sampleCount, beamCount]
                + list(curGruState.shape[2:])
            )

            return (curGruOutput, curGruState), generatedSymbolTensor

        decodedSymbolBeams, decodedStatesTupleBeams = BeamSearch(
            symbolGeneratorModel=decoderModel,
            modelStartState=(initGruOutput, initGruState),
            maxOutputLen=self.max_output_len,
            maxBeamCount=beam_count,
            sos_id=self.sos_id,
            eos_id=self.eos_id,
            outBeamCount=1,
            device=self.device,
        )

        bestBeam = decodedSymbolBeams[0]
        return self.symbolsTensor[bestBeam], bestBeam, None, None


if __name__ == "__main__":
    import pdb;pdb.set_trace()
