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
from .utils import invertPermutation, onehotencode, checkNans
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            input_dropout_p=0,
            dropout_p=0,
            use_attention=False):
        super().__init__(len(output_vocab), max_output_len, output_decoder_state_width,
                input_dropout_p, dropout_p, output_decoder_stack_depth, "gru")
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.output_decoder_stack_depth = output_decoder_stack_depth
        self.max_node_count = max_node_count
        self.max_output_len = max_output_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = output_vocab.stoi["<pad>"]

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

    def computeDimSqueezePoints(self, outputLimitsInOrder):
        """
        Compute the positions in output symbol computation, where we exclude another batch of trees from
        further consideration. We do that because the excluded output trees have their symbol computation already
        completed and need no more computation. Only used during training, when target output lengths are available.

        Input:
            outputLimitsInOrder: Length of target outputs in decreasing order.

        Output:
            dimSqueezePoints: Lit of tuples (outputIndexLimit, sampleIndexLimit)
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
        numSamples = len(outputLimitsInOrder)
        curOutputLimit = outputLimitsInOrder[-1]

        dimSqueezePoints.append((curOutputLimit, numSamples))

        for sampleLimit, outputLimit in enumerate(outputLimitsInOrder[::-1]):
            if outputLimit == curOutputLimit:
                continue

            curOutputLimit = outputLimit
            dimSqueezePoints.append((curOutputLimit, numSamples - sampleLimit))
        
        return dimSqueezePoints

    def forward(self,
            treeIndex2NodeIndex2NbrIndices,
            nodeInfoPropagatedTensor,
            targetOutput=None,
            target_lengths=None,
            teacher_forcing_ratio=0):
        duringTraining = targetOutput is not None

        if duringTraining:
            # Obtain trees in the decreasing targetOutput size order.
            targetSizeOrder = list(range(len(targetOutput)))
            targetSizeOrder.sort(key=lambda i:-target_lengths[i])

            checkNans(nodeInfoPropagatedTensor)
            nodeInfoPropagatedTensor = nodeInfoPropagatedTensor[targetSizeOrder, ...]
            checkNans(nodeInfoPropagatedTensor)
            targetOutput = targetOutput[targetSizeOrder, ...]
            targetSizes = target_lengths[targetSizeOrder]
            targetSizeOrderInverse = invertPermutation(targetSizeOrder)
            dimSqueezePoints = self.computeDimSqueezePoints(targetSizes)
        else:
            import pdb;pdb.set_trace()
            
        sampleCount = len(nodeInfoPropagatedTensor)

        outputSymbolTensors = []
        outputSymbols = []

        # Initialize cur attention by paying attention to the root node.
        # For each sample, the first node is root and hence it is the parent of itself.
        curAttention = torch.Tensor([onehotencode(self.max_node_count, 0) for _ in range(sampleCount)])
        for i in range(sampleCount):
            assert(treeIndex2NodeIndex2NbrIndices[i][0][0] == 0)

        # Current GRU state starts as all 0.
        curGruState = torch.zeros((
            self.output_decoder_stack_depth,
            sampleCount,
            self.output_decoder_state_width,))

        # Build symbol loop input.
        curSymbolTensor = torch.Tensor([onehotencode(len(self.output_vocab), self.sos_id) for _ in range(sampleCount)])

        # It is 1 at sampleIndex when teacher forcing for that index is 1, else 0.
        teacherForcingTensor = torch.Tensor(
            [
                [1 if random.random() < teacher_forcing_ratio else 0]
                for _ in range(sampleCount)
            ])

        # It is 0 at sampleIndex when teacher forcing for that index is 1, else 1.
        loopbackForcingTensor = torch.ones(1) - teacherForcingTensor

        for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
            # Clip loop variables, restricting sample indices to sampleIndexLimit.
            curAttention = curAttention[0:sampleIndexLimit, ...]
            curSymbolTensor = curSymbolTensor[0:sampleIndexLimit, ...]
            curGruState = curGruState[:, 0:sampleIndexLimit, ...]
            teacherForcingTensor = teacherForcingTensor[0:sampleIndexLimit, ...]
            loopbackForcingTensor = loopbackForcingTensor[0:sampleIndexLimit, ...]
            nodeInfoPropagatedTensor = nodeInfoPropagatedTensor[0:sampleIndexLimit, ...]
            checkNans(curAttention)
            checkNans(nodeInfoPropagatedTensor)

            while True:
                curOutputIndex = len(outputSymbolTensors)
                if curOutputIndex >= outputIndexLimit:
                    # We are crossing output index limit. So, this loop is over.
                    break
                nodeInfoToAttend = torch.bmm(
                        curAttention.view(sampleIndexLimit, 1, self.max_node_count),
                        nodeInfoPropagatedTensor
                    ).view(sampleIndexLimit, self.propagated_info_len)
                checkNans(nodeInfoToAttend)

                curGruInput = torch.cat([nodeInfoToAttend, curSymbolTensor], -1)
                curGruInput = curGruInput.view(sampleIndexLimit, 1, self.gruInputLen)
                curOutput, curGruState = self.gruCell(curGruInput, curGruState)

                # Compute next symbol.
                nextSymbolTensor, nextSymbol = self.decodeSymbol(curOutput)

                # Compute next attention.
                curAttention = self.decodeAttention(nodeInfoPropagatedTensor, curOutput)
                outputSymbols.append(nextSymbol)
                outputSymbolTensors.append(nextSymbolTensor)
                checkNans(nextSymbolTensor)

                targetSymbolTensor = torch.Tensor(
                    [
                        onehotencode(len(self.output_vocab), targetOutput[sampleIndex][curOutputIndex])
                        for sampleIndex in range(sampleIndexLimit)
                    ])

                curSymbolTensor = (
                    teacherForcingTensor * targetSymbolTensor
                    + loopbackForcingTensor * nextSymbolTensor
                )

        numSamples = dimSqueezePoints[0][1]
        outputSymbolsTransposed = []
        for _ in range(numSamples):
            outputSymbolsTransposed.append([])
        padTensor = torch.Tensor([onehotencode(len(self.output_vocab), self.pad_id)])

        for i, (curSymbolColumn, curSymbolTensorColumn) in enumerate(zip(outputSymbols, outputSymbolTensors)):
            for j, curSymbol in enumerate(curSymbolColumn):
                outputSymbolsTransposed[targetSizeOrderInverse[j]].append(curSymbol)
            if numSamples > len(curSymbolTensorColumn):
                outputSymbolTensors[i] = torch.cat(
                    [curSymbolTensorColumn] + (numSamples-len(curSymbolTensorColumn)) * [padTensor])

        outputSymbols = outputSymbolsTransposed
        outputSymbolTensors = torch.cat(
            [
                outputSymbolTensor.view([numSamples, 1, len(self.output_vocab)])
                for outputSymbolTensor in outputSymbolTensors
            ],
            1)
        outputSymbolTensors = outputSymbolTensors[targetSizeOrderInverse, ...]

        checkNans(outputSymbolTensors)
        return outputSymbolTensors, outputSymbols
