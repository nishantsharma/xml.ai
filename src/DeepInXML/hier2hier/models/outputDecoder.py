from __future__ import unicode_literals, print_function, division
import unicodedata, string, re, random, sys, copy, math
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
from hier2hier.models.hier2hierBatch import splitByToi, computeDimSqueezePoints

import torch.nn.functional as F

class OutputDecoder(ModuleBase):
    def __init__(self,
            output_vocab,
            propagated_info_len,
            attentionSubspaceVecLen,
            output_decoder_state_width,
            output_decoder_stack_depth,
            max_output_len,
            sos_id,
            eos_id,
            enableSpotlight=False,
            spotlightThreshold=0.001,
            teacher_forcing_ratio=0,
            input_dropout_p=0,
            dropout_p=0,
            device=None,
            runtests=False,
            spotlightByFormula=None,
    ):
        super().__init__(device)
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.output_decoder_stack_depth = output_decoder_stack_depth
        self.max_output_len = max_output_len
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.spotlightByFormula = spotlightByFormula

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
            spotlightThreshold,
            checkGraph=enableSpotlight,
            )
        posVecsProjectorForAttention = torch.randn(propagated_info_len, attentionSubspaceVecLen, device=self.device)
        posVecsProjectorForAttention /= math.sqrt(propagated_info_len)
        posVecsProjectorForAttention.unsqueeze(0)
        self.posVecsProjectorForAttention = nn.Parameter(posVecsProjectorForAttention)

        gruOutputProjectorForAttention = torch.randn(self.output_decoder_state_width, attentionSubspaceVecLen, device=self.device)
        gruOutputProjectorForAttention /= math.sqrt(self.output_decoder_state_width)
        gruOutputProjectorForAttention.unsqueeze(0)
        self.gruOutputProjectorForAttention = nn.Parameter(gruOutputProjectorForAttention)

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
    def forward(self,
            sampleCount,
            posNbrhoodGraphByGndtol,
            initSpotlight,
            posEncodedVecsByGndtol,
            targetOutputsByTdol,
            targetOutputLengthsByTdol,
            gndtol2Tdol,
            tdol2Toi,
            toi2Tdol,
            tensorBoardHook,
            attentionSpotLight=None,
            collectOutput=None,
            beam_count=None,
            clip_output_len=None,
            debugPack=None,
            hier2hierBatch=None,
        ):
        if debugPack is not None:
            (dataStagesToDebug, hier2hierBatch) = debugPack

        if targetOutputLengthsByTdol is not None:
            max_output_len = int(max(targetOutputLengthsByTdol))
        else:
            max_output_len = self.max_output_len

        if clip_output_len is not None:
            max_output_len=min(max_output_len, clip_output_len)

        # Dropout parts of data for robustness.e
        posEncodedVecsByGndtol = self.input_dropout(posEncodedVecsByGndtol)

        if debugPack is not None:
            # Use ndfo2Toi partition.
            dataStagesToDebug.append(splitByToi(
                posEncodedVecsByGndtol,
                hier2hierBatch.gndtol2Toi,
                hier2hierBatch.sampleCount
            ))

        if collectOutput is None:
            collectOutput = not(self.training)

        if not self.training and beam_count is not None:
            return self.beamSearch(
                sampleCount,
                posNbrhoodGraphByGndtol,
                posEncodedVecsByGndtol,
                gndtol2Tdol,
                tdol2Toi,
                beam_count)

        with blockProfiler("ProcessPreLoop"):
            if targetOutputLengthsByTdol is not None:
                # Get inverse of the TL order.
                dimSqueezePoints = computeDimSqueezePoints(targetOutputLengthsByTdol)
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
            outputSymbolsByTdolList = [ curSymbolTensor ]
            if collectOutput:
                outputSymbols = [ [ self.sos_id for _ in range(sampleCount) ] ]
            else:
                outputSymbols = None

            attnReadyVecsByGndtol = torch.matmul(
                posEncodedVecsByGndtol.unsqueeze(1),
                self.posVecsProjectorForAttention,
            ).squeeze(1)

        if debugPack is not None:
            # Use ndfo2Toi partition.
            dataStagesToDebug.append(splitByToi(
                posEncodedVecsByGndtol,
                hier2hierBatch.gndtol2Toi,
                hier2hierBatch.sampleCount
            ))

        with blockProfiler("Loop"):
            for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
                if curGruOutput.shape[0] != sampleIndexLimit:
                    with blockProfiler("ReShaping"):
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
                    symbolIndex = len(outputSymbolsByTdolList)
                    if symbolIndex == outputIndexLimit:
                        # We are crossing output index limit. So, this loop is over.
                        break

                    if debugPack is not None:
                        # Use ndfo2Toi partition.
                        dataStagesToDebug.append(splitByToi(
                            curGruOutput,
                            hier2hierBatch.tdol2Toi,
                            hier2hierBatch.sampleCount,
                            prefix="{0}@".format(symbolIndex),
                        ))

                    # Compute next attention.
                    with blockProfiler("ATTENTION-DECODE"):
                        # Down sample curGruOutput for efficiency.
                        attnReadyGruOutput = torch.matmul(
                            curGruOutput.unsqueeze(1),
                            self.gruOutputProjectorForAttention,
                        ).squeeze(1)

                        if self.spotlightByFormula is None:
                            (
                                sli2Gndtol,
                                attnReadyInfoCollapsedByTdol
                            ) = self.attentionSpotlight(
                                posNbrhoodGraphByGndtol,
                                attnReadyVecsByGndtol,
                                posEncodedVecsByGndtol,
                                sli2Gndtol,
                                gndtol2Tdol,
                                attnReadyGruOutput,
                                sampleIndexLimit,
                                tensorBoardHook,
                                beamMode=False,
                                debugPack=None,#debugPack,
                            )
                        else:
                            sli2Gndtol = self.spotlightByFormula(
                                hier2hierBatch,
                                sampleIndexLimit,
                                symbolIndex,
                            )
                            attnReadyInfoCollapsedByTdol  = posEncodedVecsByGndtol[sli2Gndtol]
                    # print("Selected indices: {0}".format(sli2Gndtol.shape[0]))

                    if debugPack is not None:
                        # Use ndfo2Toi partition.
                        dataStagesToDebug.append(splitByToi(
                            attnReadyInfoCollapsedByTdol,
                            hier2hierBatch.tdol2Toi,
                            hier2hierBatch.sampleCount,
                            prefix="{0}@".format(symbolIndex),
                        ))

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
                        curGruState = curGruState.contiguous()

                    if debugPack is not None:
                        # Use ndfo2Toi partition.
                        dataStagesToDebug.append(splitByToi(
                            curGruState.permute(1,0,2),
                            hier2hierBatch.tdol2Toi,
                            hier2hierBatch.sampleCount,
                            prefix="{0}@".format(symbolIndex),
                        ))

                    # Cycle RNN state.
                    with blockProfiler("GRUCELL2"):
                        tensorBoardHook.add_histogram("OutputDecoderGru.Input", curGruInput)
                        tensorBoardHook.add_histogram("OutputDecoderGru.State", curGruState)
                        curGruOutput, curGruState = self.gruCell(curGruInput, curGruState)
                        curGruOutput = curGruOutput.view(sampleIndexLimit, self.output_decoder_state_width)

                    # Compute next symbol.
                    with blockProfiler("SYMBOL-DECODE"):
                        generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
                        generatedSymbolTensor = generatedSymbolTensor.view(sampleIndexLimit, len(self.output_vocab))
                        outputSymbolsByTdolList.append(generatedSymbolTensor)

                    if debugPack is not None:
                        # Use ndfo2Toi partition.
                        dataStagesToDebug.append(splitByToi(
                            generatedSymbolTensor,
                            hier2hierBatch.tdol2Toi,
                            hier2hierBatch.sampleCount,
                            prefix="{0}@".format(symbolIndex),
                        ))

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
            symbolColumnsCount = len(outputSymbolsByTdolList)
            if collectOutput:
                outputSymbolsTransposed = [[] for _ in range(sampleCount)]
                for curSymbolColumn in outputSymbols:
                    for j, curSymbol in enumerate(curSymbolColumn):
                        outputSymbolsTransposed[tdol2Toi[j]].append(curSymbol)
                outputSymbols = outputSymbolsTransposed

            checkNans(outputSymbolsByTdolList)
            return outputSymbolsByTdolList, outputSymbols, teacherForcedSelections

    @methodProfiler
    def beamSearch(self,
                sampleCount,
                posNbrhoodGraphByGndtol,
                posEncodedVecsByGndtol,
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
                posEncodedVecsByGndtol,
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
