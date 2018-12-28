from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys, copy
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .baseRNN import BaseRNN
from .attention import Attention
from hier2hier.util import (onehotencode, checkNans, invertPermutation, blockProfiler,
                            methodProfiler, lastCallProfile)
import torch.nn.functional as F

class OutputDecoder(BaseRNN):
    def __init__(self,
            output_vocab,
            propagated_info_len,
            output_decoder_state_width,
            output_decoder_stack_depth,
            max_node_count,
            max_output_len,
            sos_id,
            eos_id,
            teacher_forcing_ratio=0,
            input_dropout_p=0,
            dropout_p=0,
            use_attention=False,
            device=None,
            runtests=False):
        super().__init__(len(output_vocab), max_output_len, output_decoder_state_width,
                input_dropout_p, dropout_p, output_decoder_stack_depth, "gru", device)
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.output_decoder_stack_depth = output_decoder_stack_depth
        self.max_node_count = max_node_count
        self.max_output_len = max_output_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = output_vocab.stoi["<pad>"]
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gruInputLen = propagated_info_len + len(output_vocab)
        self.gruCell = self.rnn_cell(
            self.gruInputLen,
            self.output_decoder_state_width,
            self.output_decoder_stack_depth,
            batch_first=True,
            dropout=dropout_p)

        self.decode_function = F.log_softmax
        self.init_input = None

        self.symbolsTensor = torch.eye(len(output_vocab), device=self.device)

        # self.embedding = nn.Embedding(self.output_size, self.output_size)

        # Network for attention decoding.
        self.attentionPreDecoder = nn.Linear(output_decoder_state_width + propagated_info_len, 1)
        self.attentionDecoder = nn.Softmax(dim=-1)

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

    @torch.no_grad()
    def test_forward(self,
            treeIndex2NodeIndex2NbrIndices,
            nodeInfoPropagatedTensor,
            targetOutput=None,
            targetLengths=None,
            teacherForcedSelections=None):

        treeCount = len(nodeInfoPropagatedTensor)
        duringTraining = targetOutput is not None

        outputSymbolsTensor = []
        outputSymbols = []
        for treeIndex in range(treeCount):
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
            if not duringTraining or teacherForcedSelections is None:
                teacherForced = False
            else:
                teacherForced = teacherForcedSelections[treeIndex]

            treeOutputSymbolsTensor = [ ]
            treeOutputSymbols = [ ]

            nodeInfosPropagated = nodeInfoPropagatedTensor[treeIndex]
            maxSymbolIndex = -1
            for symbolIndex in range(self.max_output_len):
                # Compute next attention.
                curAttention = self.decodeAttention(
                    nodeInfosPropagated.view(1, 1, -1),
                    curGruOutput)
                
                # Compute node info summary to use in computation.
                propagatedNodeInfoToAttend = torch.mm(
                        curAttention,
                        nodeInfosPropagated,
                    ).view(1, -1)

                # Build GRU Input.
                curGruInput = torch.cat([propagatedNodeInfoToAttend, curSymbolTensor], -1)
                curGruInput = curGruInput.view(1, 1, self.gruInputLen)

                # For first iteration, we also need to build GRU State.
                if curGruState is None:
                    curGruState = self.buildInitGruState(propagatedNodeInfoToAttend)
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
                if not duringTraining:
                    generatedSymbol = int(generatedSymbolTensor.topk(1))
                    treeOutputSymbols.append(generatedSymbol)

                # Loop completion checks.
                if ((duringTraining and symbolIndex == targetLengths[treeIndex]-1)
                        or (not duringTraining and generatedSymbol == self.eos_id)):
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
                        int(targetOutput[treeIndex, symbolIndex]))
                    curSymbolTensor = torch.tensor([curSymbolTensor])
                else:
                    curSymbolTensor = generatedSymbolTensor

            treeOutputSymbolsTensor = torch.cat(treeOutputSymbolsTensor)
            outputSymbolsTensor.append(treeOutputSymbolsTensor)
            outputSymbols.append(treeOutputSymbols)

        outputSymbolsTensor = torch.cat(outputSymbolsTensor).view(
            treeCount,
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
            treeIndex2NodeIndex2NbrIndices,
            nodeInfoPropagatedTensor,
            targetOutput,
            targetLengths,
            tensorBoardHook,
        ):
        sampleCount = len(nodeInfoPropagatedTensor)
        with blockProfiler("ProcessPreLoop"):
            duringTraining = targetOutput is not None
            if duringTraining:
                # Obtain trees in the decreasing targetOutput size order.
                targetLengthsOrder = list(range(len(targetOutput)))
                targetLengthsOrder.sort(key=lambda i:-targetLengths[i])
                targetLengthsOrder = torch.tensor(targetLengthsOrder, dtype=torch.long, device=self.device)

                # Re-arrange nodeInfoPropagatedTensor in target lengths order.
                nodeInfoPropagatedTensor = torch.index_select(
                    nodeInfoPropagatedTensor,
                    0,
                    targetLengthsOrder)

                # Re-arrange targetOutput and targetLengths in target lengths order.
                targetOutput = torch.index_select(targetOutput, 0, targetLengthsOrder)
                targetLengths = torch.index_select(targetLengths, 0, targetLengthsOrder)

                # Get the inverse of the TL order.
                targetLengthsOrderInverse = torch.tensor(
                    invertPermutation(targetLengthsOrder),
                    dtype=torch.long,
                    device=self.device,
                )

                # Get inverse of the TL order.
                dimSqueezePoints = self.computeDimSqueezePoints(targetLengths)
            else:
                # Target lengths are unknown. Emulate with what we have.
                targetLengthsOrderInverse = torch.tensor(
                    list(range(self.max_output_len)),
                    dtype=torch.long)
                dimSqueezePoints = [(self.max_output_len, sampleCount)]

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
            teacherForcedSelections = [
                True if random.random() < self.teacher_forcing_ratio else False
                for n in range(sampleCount)
            ]

            # Init output vars.
            outputSymbolTensors = [ ]
            outputSymbols = None if duringTraining else [ ]

        with blockProfiler("Loop"):
            for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
                if nodeInfoPropagatedTensor.shape[0] != sampleIndexLimit:
                    # Clip loop variables, restricting sample indices to sampleIndexLimit.
                    curAttention = curAttention.narrow(0, 0, sampleIndexLimit)
                    curSymbolTensor = curSymbolTensor.narrow(0, 0, sampleIndexLimit)
                    curGruState = curGruState.narrow(1, 0, sampleIndexLimit)
                    curGruOutput = curGruOutput.narrow(0, 0, sampleIndexLimit)
                    nodeInfoPropagatedTensor = nodeInfoPropagatedTensor.narrow(0, 0, sampleIndexLimit)
                    if targetOutput is not None:
                        targetOutput = targetOutput.narrow(0, 0, sampleIndexLimit)

                # Logic for applying 
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
                        curAttention = self.decodeAttention(nodeInfoPropagatedTensor, curGruOutput)

                    with blockProfiler("BMM"):
                        propagatedNodeInfoToAttend = torch.bmm(
                                curAttention.view(sampleIndexLimit, 1, self.max_node_count),
                                nodeInfoPropagatedTensor,
                            ).view(sampleIndexLimit, self.propagated_info_len)

                    with blockProfiler("CATVIEW"):
                        curGruInput = torch.cat([propagatedNodeInfoToAttend, curSymbolTensor], -1)
                        curGruInput = curGruInput.view(sampleIndexLimit, 1, self.gruInputLen)

                    if curGruState is None:
                        with blockProfiler("BuildInitGruState"):
                            curGruState = self.buildInitGruState(propagatedNodeInfoToAttend)
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

                    # Compute next symbol.
                    with blockProfiler("SYMBOL-DECODE"):
                        generatedSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curGruOutput))
                        generatedSymbolTensor = generatedSymbolTensor.view(sampleIndexLimit, len(self.output_vocab))
                        outputSymbolTensors.append(generatedSymbolTensor)

                    # Compute next symbol list.
                    if not duringTraining:
                        with blockProfiler("TOPK"):
                            generatedSymbol = [int(symbol) for symbol in generatedSymbolTensor.topk(1)[1].view(sampleIndexLimit)]
                            outputSymbols.append(generatedSymbol)


                    ###########################################################################
                    ##### CURRENT LOOP ITERATION COMPLETE. Now preping for next iteration #####
                    ###########################################################################

                    if duringTraining and symbolIndex == targetOutput.shape[1]-1:
                        # We are crossing output index limit. So, this loop is over.
                        break

                    if duringTraining:
                        with blockProfiler("TEACHER-FORCING"):
                            targetSymbolsTensor = self.symbolsTensor[
                                targetOutput[..., symbolIndex]
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
            if not duringTraining:
                outputSymbolsTransposed = [[] for _ in range(sampleCount)]
                for curSymbolColumn in outputSymbols:
                    for j, curSymbol in enumerate(curSymbolColumn):
                        outputSymbolsTransposed[targetLengthsOrderInverse[j]].append(curSymbol)
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
            if duringTraining:
                outputSymbolTensors = torch.index_select(outputSymbolTensors, 0, targetLengthsOrderInverse)

            if self.runtests:
                padTensorColumn = torch.cat([padTensor.view(1, 1, -1) for _ in range(sampleCount)])
                decoderTestTensors = torch.cat(
                    [
                        outputSymbolTensors
                    ]
                    +
                    [
                        padTensorColumn
                        for _ in range(self.max_output_len - symbolColumnsCount)
                    ],
                    -2
                )
            else:
                decoderTestTensors = None

            checkNans(outputSymbolTensors)
            return outputSymbolTensors, outputSymbols, teacherForcedSelections, decoderTestTensors
