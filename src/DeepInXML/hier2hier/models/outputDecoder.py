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
            device=None):
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

        # self.embedding = nn.Embedding(self.output_size, self.output_size)

        # Network for attention decoding.
        self.attentionPreDecoder = nn.Linear(output_decoder_state_width + propagated_info_len, 1)
        self.attentionDecoder = nn.Softmax(dim=-1)

        # Network for symbol decoding.
        self.symbolPreDecoder = nn.Linear(output_decoder_state_width, len(output_vocab))
        self.symbolDecoder = nn.Softmax(dim=-1)
        self.output_vocab = output_vocab

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
            targetOutput=None,
            target_lengths=None):
        try:
            symbolsTensor = self.symbolsTensor
        except:
            n = len(self.output_vocab)
            symbolsTensor = torch.tensor([
                onehotencode(n, i)
                for i in range(n)
            ], device=self.device)

            self.symbolsTensor = symbolsTensor

        sampleCount = len(nodeInfoPropagatedTensor)
        with blockProfiler("ProcessPreLoop"):

            duringTraining = targetOutput is not None
            if duringTraining:
                # Obtain trees in the decreasing targetOutput size order.
                targetSizeOrder = list(range(len(targetOutput)))
                targetSizeOrder.sort(key=lambda i:-target_lengths[i])
                targetSizeOrder = torch.tensor(targetSizeOrder, dtype=torch.long)

                nodeInfoPropagatedTensor = torch.index_select(
                    nodeInfoPropagatedTensor,
                    0,
                    targetSizeOrder)

                targetOutput = torch.index_select(targetOutput, 0, targetSizeOrder)
                targetSizes = torch.index_select(target_lengths, 0, targetSizeOrder)
                targetSizeOrderInverse = torch.tensor(
                    invertPermutation(targetSizeOrder),
                    dtype=torch.long)
                dimSqueezePoints = self.computeDimSqueezePoints(targetSizes)
            else:
                targetSizeOrderInverse = torch.tensor(
                    list(range(self.max_output_len)),
                    dtype=torch.long)
                dimSqueezePoints = [(self.max_output_len, sampleCount)]

            outputSymbolTensors = []
            outputSymbols = None if duringTraining else []

            # Initialize cur attention by paying attention to the root node.
            # For each sample, the first node is root and hence it is the parent of itself.
            curAttention = torch.tensor([onehotencode(self.max_node_count, 0) for _ in range(sampleCount)],
                                        device=self.device)
            for i in range(sampleCount):
                assert(treeIndex2NodeIndex2NbrIndices[i][0][0] == 0)

            # Current GRU state starts as all 0.
            curGruState = torch.zeros((
                self.output_decoder_stack_depth,
                sampleCount,
                self.output_decoder_state_width,), device=self.device)

            # Build symbol loop input.
            curSymbolTensor = torch.tensor([
                onehotencode(len(self.output_vocab), self.sos_id)
                for _ in range(sampleCount)],
                device=self.device)

            # It is 1 at sampleIndex when teaching is being forced, else 0.
            teacherForcedSelections = [
                1 if random.random() < self.teacher_forcing_ratio else 0
                for n in range(sampleCount)
            ]

            print("Dim Squeeze points", dimSqueezePoints)
        with blockProfiler("Loop"):
            for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
                # Clip loop variables, restricting sample indices to sampleIndexLimit.
                curAttention = curAttention.narrow(0, 0, sampleIndexLimit)
                curSymbolTensor = curSymbolTensor.narrow(0, 0, sampleIndexLimit)
                curGruState = curGruState.narrow(1, 0, sampleIndexLimit)
                nodeInfoPropagatedTensor = nodeInfoPropagatedTensor.narrow(0, 0, sampleIndexLimit)
                if targetOutput is not None:
                    targetOutput = targetOutput.narrow(0, 0, sampleIndexLimit)
                teacherForcingApplicator = torch.tensor(
                    [
                        n + teacherForcedSelections[n] * sampleIndexLimit
                        for n in range(sampleIndexLimit)
                    ],
                    dtype=torch.long
                )

                while True:
                    curOutputIndex = len(outputSymbolTensors)
                    if curOutputIndex >= outputIndexLimit:
                        # We are crossing output index limit. So, this loop is over.
                        break
                    with blockProfiler("BMM"):
                        nodeInfoToAttend = torch.bmm(
                                curAttention.view(sampleIndexLimit, 1, self.max_node_count),
                                nodeInfoPropagatedTensor
                            ).view(sampleIndexLimit, self.propagated_info_len)

                    with blockProfiler("CATVIEW"):
                        curGruInput = torch.cat([nodeInfoToAttend, curSymbolTensor], -1)
                        curGruInput = curGruInput.view(sampleIndexLimit, 1, self.gruInputLen)

                    with blockProfiler("GRUCELL1"):
                        curGruStateContiguous = curGruState.contiguous()
                        
                    with blockProfiler("GRUCELL2"):
                        curOutput, curGruState = self.gruCell(curGruInput, curGruStateContiguous)

                    # Compute next symbol.
                    with blockProfiler("SYMBOL-DECODE"):
                        nextSymbolTensor = self.symbolDecoder(self.symbolPreDecoder(curOutput))
                        nextSymbolTensor = nextSymbolTensor.view(len(curOutput), len(self.output_vocab))

                    # Compute next symbol list.
                    if not duringTraining:
                        with blockProfiler("TOPK"):
                            nextSymbol = [int(symbol) for symbol in nextSymbolTensor.topk(1)[1].view(len(curOutput))]
                            outputSymbols.append(nextSymbol)

                    # Compute next attention.
                    with blockProfiler("ATTENTION-DECODE"):
                        curAttention = self.decodeAttention(nodeInfoPropagatedTensor, curOutput)
                        outputSymbolTensors.append(nextSymbolTensor)

                    if duringTraining:
                        with blockProfiler("TEACHER-FORCING"):
                            targetSymbolsTensor = symbolsTensor[
                                targetOutput[..., curOutputIndex]
                            ]

                            concatedSymbolTensors = torch.cat([
                                nextSymbolTensor, # Regular input.
                                targetSymbolsTensor, # Teaching forced.
                            ])

                            curSymbolTensor = torch.index_select(concatedSymbolTensors, 0, teacherForcingApplicator)
                    else:
                        curSymbolTensor = nextSymbolTensor


        with blockProfiler("ProcessPostLoop"):
            if not duringTraining:
                outputSymbolsTransposed = [[] for _ in range(sampleCount)]
                for curSymbolColumn in outputSymbols:
                    for j, curSymbol in enumerate(curSymbolColumn):
                        outputSymbolsTransposed[targetSizeOrderInverse[j]].append(curSymbol)
                outputSymbols = outputSymbolsTransposed

            # Pad symbol tensor columns to include pad symbols.
            padTensor = torch.tensor([onehotencode(len(self.output_vocab), self.pad_id)], device=self.device)
            for i, curSymbolTensorColumn in enumerate(outputSymbolTensors):
                if sampleCount > len(curSymbolTensorColumn):
                    outputSymbolTensors[i] = torch.cat(
                        [curSymbolTensorColumn] + (sampleCount-len(curSymbolTensorColumn)) * [padTensor])

            outputSymbolTensors = torch.cat(
                [
                    outputSymbolTensor.view([sampleCount, 1, len(self.output_vocab)])
                    for outputSymbolTensor in outputSymbolTensors
                ],
                1)

            # During training, we permute the columns 
            if duringTraining:
                outputSymbolTensors = torch.index_select(outputSymbolTensors, 0, targetSizeOrderInverse)

            checkNans(outputSymbolTensors)
            return outputSymbolTensors, outputSymbols
