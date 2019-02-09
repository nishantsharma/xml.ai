"""
This module implements an attentional decoder for transforming an input XML to output.
"""
from __future__ import unicode_literals, print_function, division
import unicodedata, string, re, random, sys, copy, math
from orderedattrdict import AttrDict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moduleBase import ModuleBase
from .attentionSpotlight import AttentionSpotlight
from .beamSearch import BeamSearch
from .hier2hierBatch import splitByToi, computeDimSqueezePoints
from .symbolDecoder import SymbolDecoder

from hier2hier.util import (onehotencode, checkNans, blockProfiler,
                            methodProfiler, lastCallProfile)

import torch.nn.functional as F

class OutputDecoder(ModuleBase):
    def __init__(self,
            schemaVersion,
            tgtVocabs,
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
            useSrcPtr=True,
            device=None,
            runtests=False,
            spotlightByFormula=None,
    ):
        super().__init__(schemaVersion, device)
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.output_decoder_stack_depth = output_decoder_stack_depth
        self.max_output_len = max_output_len
        self.attentionSubspaceVecLen = attentionSubspaceVecLen
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.spotlightByFormula = spotlightByFormula

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = tgtVocabs.all.stoi["<pad>"]
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gruInputLen = propagated_info_len + len(tgtVocabs.all)

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.gruCell = nn.GRU(
            self.gruInputLen,
            self.output_decoder_state_width,
            self.output_decoder_stack_depth,
            batch_first=True,
            dropout=dropout_p)
        self.symbolsTensor = nn.Parameter(torch.eye(len(tgtVocabs.all), device=self.device), requires_grad=False)

        # Module for decoding attention and moving spotlight.
        self.attentionSpotlight = AttentionSpotlight(
            schemaVersion,
            spotlightThreshold,
            checkGraph=enableSpotlight,
            device=self.device,
            )
        posVecsProjectorForAttention = torch.randn(propagated_info_len, attentionSubspaceVecLen, device=self.device)
        posVecsProjectorForAttention /= math.sqrt(propagated_info_len)
        posVecsProjectorForAttention.unsqueeze(0)
        self.posVecsProjectorForAttention = nn.Parameter(posVecsProjectorForAttention)

        gruOutputProjectorForAttention = torch.randn(self.output_decoder_state_width, attentionSubspaceVecLen, device=self.device)
        gruOutputProjectorForAttention /= math.sqrt(self.output_decoder_state_width)
        gruOutputProjectorForAttention.unsqueeze(0)
        self.gruOutputProjectorForAttention = nn.Parameter(gruOutputProjectorForAttention)
        self.tgtVocabs = tgtVocabs
        self.runtests = runtests

        # Parameters required for loop initialization.
        self.initGruOutput = nn.Parameter(torch.zeros(self.output_decoder_state_width, ))

        # Network for symbol decoding.
        if schemaVersion == 0:
            self.symbolPreDecoder = nn.Linear(output_decoder_state_width, len(tgtVocabs.all))
            self.symbolDecoder = nn.Softmax(dim=-1)
            self.buildInitGruState = nn.Linear(
                self.propagated_info_len,
                self.output_decoder_stack_depth * self.output_decoder_state_width)
        else:
            self.symbolDecoder = SymbolDecoder(
                schemaVersion,
                useSrcPtr,
                tgtVocabs,
                output_decoder_state_width,
                device,
            )
            stateLength = self.output_decoder_stack_depth * self.output_decoder_state_width
            self.buildInitGruState = nn.Sequential(OrderedDict([
                ("Linear1", nn.Linear(self.propagated_info_len, stateLength)),
                ("Selu1", nn.SELU()),
                ("Linear2", nn.Linear(stateLength, stateLength)),
                ("Selu2", nn.SELU()),
                ("Linear3", nn.Linear(stateLength, stateLength)),
                ("Selu3", nn.SELU()),
            ]))

    def reset_parameters(self, device):
        self.gruCell.reset_parameters()
        self.attentionSpotlight.reset_parameters(device)
        nn.init.normal_(self.initGruOutput)
        self.buildInitGruState.reset_parameters()

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 1:
            # Migrate to schema 1.
            symbolDecoder = SymbolDecoder(
                schemaVersion,
                True,
                self.tgtVocabs,
                self.output_decoder_state_width,
                self.device,
            )
            symbolDecoder.nonPointingShaper = self.symbolPreDecoder
            symbolDecoder.softMax = self.symbolDecoder
            self.symbolDecoder = symbolDecoder
            del self.symbolPreDecoder

            # Beef up initial GRU state builder.
            # Use the ealier state.
            stateLength = self.output_decoder_stack_depth * self.output_decoder_state_width
            self.buildInitGruState = nn.Sequential(OrderedDict([
                ("Linear1", self.buildInitGruState),
                ("Selu1", nn.SELU()),
                ("Linear2", nn.Linear(stateLength, stateLength)),
                ("Selu2", nn.SELU()),
                ("Linear3", nn.Linear(stateLength, stateLength)),
                ("Selu3", nn.SELU()),
            ]))
            for linearSubModule in [self.buildInitGruState.Linear2, self.buildInitGruState.Linear3]:
                linearSubModule.weights = nn.eye(stateLength, device=self.device)
                linearSubModule.bias = nn.zeros(stateLength, device=self.device)
        else:
            super().singleStepSchema(schemaVersion)

    @methodProfiler
    def forward(self,
            sampleCount,
            posNbrhoodGraphByGndtol,
            initSpotlight,
            srcSymbolsByGndtol,
            posEncodedVecsByGndtol,
            targetOutputsByTdol,
            targetOutputLengthsByTdol,
            gndtol2Tdol,
            goi2Gndtol,
            tdol2Toi,
            toi2Tdol,
            tensorBoardHook,
            attentionSpotLight=None,
            collectOutput=None,
            beam_count=None,
            clip_output_len=None,
            debugAttention=False,
            dataDebugHook=None,
            hier2hierBatch=None,
        ):

        if targetOutputLengthsByTdol is not None:
            max_output_len = int(max(targetOutputLengthsByTdol))
        else:
            max_output_len = self.max_output_len

        if clip_output_len is not None:
            max_output_len=min(max_output_len, clip_output_len)

        # Dropout parts of data for robustness.
        posEncodedVecsByGndtol = self.input_dropout(posEncodedVecsByGndtol)
        dataDebugHook(posEncodedVecsByGndtol, "gndtol")

        if collectOutput is None:
            collectOutput = not(self.training)

        attnReadyVecsByGndtol = torch.matmul(
            posEncodedVecsByGndtol.unsqueeze(1),
            self.posVecsProjectorForAttention,
        ).squeeze(1)

        if not self.training and beam_count is not None:
            return self.beamSearch(
                sampleCount,
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                initSpotlight,
                posEncodedVecsByGndtol,
                gndtol2Tdol,
                tdol2Toi,
                tensorBoardHook,
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
                    onehotencode(len(self.tgtVocabs.all), curSymbol)
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

            if debugAttention:
                debugAttentionFactorsImg = []

        dataDebugHook(posEncodedVecsByGndtol, "gndtol")

        with blockProfiler("Loop"):
            for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
                if curGruOutput.shape[0] != sampleIndexLimit:
                    with blockProfiler("ReShaping"):
                        # Clip loop variables, restricting sample indices to sampleIndexLimit.
                        curSymbolTensor = curSymbolTensor.narrow(0, 0, sampleIndexLimit)
                        curGruState = curGruState.narrow(1, 0, sampleIndexLimit)
                        curGruOutput = curGruOutput.narrow(0, 0, sampleIndexLimit)

                        # Shorten sli2Gndtol by dropping all trees
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

                    dataDebugHook(curGruOutput, "tdol", prefix="{0}@".format(symbolIndex))

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
                                attnReadyInfoCollapsedByTdol,
                                attnFactorsBySli, 
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
                                debugAttention=debugAttention,
                                dataDebugHook=None,#dataDebugHook
                            )
                            if debugAttention: 
                               attnFactorsByGndtol = torch.zeros(
                                       posNbrhoodGraphByGndtol[1].shape,
                                       device=self.device,
                                       requires_grad=False)
                               attnFactorsByGndtol[sli2Gndtol] = attnFactorsBySli.squeeze(-1)
                               debugAttnFactorsByGoi = attnFactorsByGndtol[goi2Gndtol]
                               debugAttentionFactorsImg.append(debugAttnFactorsByGoi.unsqueeze(0)) 
                        else:
                            sli2Gndtol = self.spotlightByFormula(
                                hier2hierBatch,
                                sampleIndexLimit,
                                symbolIndex,
                            )
                            attnReadyInfoCollapsedByTdol  = posEncodedVecsByGndtol[sli2Gndtol]
                    # print("Selected indices: {0}".format(sli2Gndtol.shape[0]))

                    dataDebugHook(attnReadyInfoCollapsedByTdol, "tdol", prefix="{0}@".format(symbolIndex))

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

                    dataDebugHook(curGruState.permute(1,0,2), "tdol", prefix="{0}@".format(symbolIndex))

                    # Cycle RNN state.
                    with blockProfiler("GRUCELL2"):
                        tensorBoardHook.add_histogram("OutputDecoderGru.Input", curGruInput)
                        tensorBoardHook.add_histogram("OutputDecoderGru.State", curGruState)
                        curGruOutput, curGruState = self.gruCell(curGruInput, curGruState)
                        curGruOutput = curGruOutput.view(sampleIndexLimit, self.output_decoder_state_width)

                    # Compute next symbol.
                    with blockProfiler("SYMBOL-DECODE"):
                        if self.schemaVersion == 0:
                            generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
                        else:
                            generatedSymbolTensor = self.symbolDecoder(
                                curGruOutput,
                                srcSymbolsByGndtol[sli2Gndtol],
                                gndtol2Tdol[sli2Gndtol],
                                attnFactorsBySli,
                                sampleIndexLimit,
                                tensorBoardHook,
                            )
                        outputSymbolsByTdolList.append(generatedSymbolTensor)

                    dataDebugHook(generatedSymbolTensor, "tdol", prefix="{0}@".format(symbolIndex))

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

        if debugAttention: 
           debugAttentionFactorsImg = torch.cat(debugAttentionFactorsImg )
           tensorBoardHook.add_image("debugAttentionFactorsImg", debugAttentionFactorsImg, dataformats="HW")

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
                attnReadyVecsByGndtol,
                initSpotlight,
                posEncodedVecsByGndtol,
                gndtol2Tdol,
                tdol2Toi,
                tensorBoardHook,
                beam_count,
    ):
        """
        This module implements beam search over all attention ready positions(text indices as 
        well as XML nodes and attributes) in an XML file. 

        Inputs:
            sampleCount: Number of samples for which search is being undertaken.
            posNbrhoodGraphByGndtol:
                Attention ready positions are represented by this graph.
                Tuple of two tensors, (nbrs, nbrCounts).
                At index gndtol, nbrs[gndtol] gives GNDTOL indices of all neighbors of graph
                    node at index gndtol.
                nbrCounts[gndtol] gives number of neighbors of node at GNDTOL index gndtol.
            attnReadyVecsByGndtol:
                Each vector attnReadyVecsByGndtol[gndtol] at index gndtol gives the vector
                which can be dot-ted with a vector derived from decoder state to determine
                attention factors.
            initSpotlight:
                Attention spotlight is a subset of graph nodes which we are currently attending to.
                For each symbol decoded, they change.
                initSpotlight is used as the initial subset.
            posEncodedVecsByGndtol
                Each vector attnReadyVecsByGndtol[gndtol] at index gndtol helps determine the
                next symbol we end up decoding. Similar to attnReadyVecsByGndtol, but used
                for decoding instead of attention.
        """
        # Initial spotlight indices.
        sli2Gndtol = initSpotlight 

        bigSampleCount = sampleCount * beam_count
        # Build first GruOutput to kickstart the loop.
        initGruOutput = torch.cat([self.initGruOutput.unsqueeze(0) for _ in range(sampleCount)])

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

            # Down sample curGruOutput for efficiency.
            attnReadyPrevGruOutput = torch.matmul(
                prevGruOutput.view(-1, self.output_decoder_state_width).unsqueeze(1),
                self.gruOutputProjectorForAttention,
            ).view(sampleCount, beamCount, -1)

            nonlocal sli2Gndtol

            (
                sli2Gndtol,
                attnReadyInfoCollapsedByTdol,
                attnFactorsByGndtol, 
            ) = self.attentionSpotlight(
                posNbrhoodGraphByGndtol,
                attnReadyVecsByGndtol,
                posEncodedVecsByGndtol,
                sli2Gndtol,
                gndtol2Tdol,
                attnReadyPrevGruOutput,
                sampleCount,
                tensorBoardHook,
                beamMode=True,
            )

            # sampleCount X beamCount X * -> bigSampleCount X *
            attnReadyInfoCollapsedByTdol = attnReadyInfoCollapsedByTdol.contiguous()
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

            # Find input symbol for all graph positions(Pre-Computed).
            # Use Attention to collapse that vec into TDOL.
            # Use curGruState to compute weight of direct copy versus derived value.
            # Combine with symbolDecoder and symbolPreDecoder to obtain the expected symbol. 
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
        return self.symbolsTensor[bestBeam], bestBeam, None


if __name__ == "__main__":
    import pdb;pdb.set_trace()
