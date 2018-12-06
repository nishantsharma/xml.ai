from __future__ import unicode_literals, print_function, division
import unicodedata
import string, re, random, sys
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .baseRNN import BaseRNN
from .attention import Attention
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
        symbol = self.output_vocab.itos[int(symbolTensor.topk(1)[1])]
        return symbolTensor, symbol

    def decodeAttention(self, nodeInfosPropagated, curGruOutput):
        curGruOutput = curGruOutput.view(self.output_decoder_state_width).repeat(self.max_node_count, 1)
        attentionInput = torch.cat([nodeInfosPropagated, curGruOutput], -1)
        return self.attentionDecoder(self.attentionPreDecoder(attentionInput).view(self.max_node_count))

    def forward(self,
            treeIndex2NodeIndex2NbrIndices,
            nodeInfoPropagatedTensor,
            targetOutput,
            teacher_forcing_ratio=0):
        sampleCount = len(nodeInfoPropagatedTensor)

        # Recurrent loop.
        outputs = []
        import pdb;pdb.set_trace()
        for sampleIndex in range(sampleCount):
            # Initialize cur attention by paying attention to the root node.
            # For each sample, the first node is root and hence it is the parent of itself.
            curAttention = torch.zeros((self.max_node_count,))
            assert(treeIndex2NodeIndex2NbrIndices[sampleIndex][0][0] == 0)
            curAttention[0] = 1

            # Current GRU state starts as all 0.
            curGruState = torch.zeros((self.output_decoder_stack_depth, 1, self.output_decoder_state_width,))

            # Build symbol loop input.
            curSymbolTensor = torch.zeros([len(self.output_vocab)])
            curSymbolTensor[self.sos_id] = 1

            # Teach forcing is selecated on a per sample basis.
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            # Symbol loop for the current sample.
            sampleOutput = []
            symbolIndex = 0
            while True:
                nodeInfoToAttend = torch.mm(
                    curAttention.view(1, self.max_node_count),
                    nodeInfoPropagatedTensor[sampleIndex])

                # We are doing it in loop and use a batch of 1 and sequence of 1. VERY SLOW INDEED.
                nodeInfoToAttend = nodeInfoToAttend.view(1, 1, self.propagated_info_len)
                curGruInput = torch.cat([nodeInfoToAttend, curSymbolTensor.view(1, 1, len(self.output_vocab))], -1)
                curOutput, curGruState = self.gruCell(curGruInput, curGruState)

                # Compute next symbol.
                nextSymbolTensor, nextSymbol = self.decodeSymbol(curOutput)

                # Compute next attention.
                curAttention = self.decodeAttention(nodeInfoPropagatedTensor[sampleIndex], curOutput)

                if nextSymbol != self.eos_id:
                    sampleOutput.append(nextSymbol)

                if use_teacher_forcing:
                    curSymbolTensor = torch.Tensor([len(self.output_vocab)])
                    curSymbolTensor[targetOutput[sampleIndex, symbolIndex]] = 1
                    curSymbol = targetOutput[sampleIndex, symbolIndex]
                else:
                    curSymbolTensor = nextSymbolTensor
                    curSymbol = nextSymbol
                
                symbolIndex += 1
                if curSymbol == self.eos_id:
                    break
    
            outputs.append(sampleOutput)
        return outputs
