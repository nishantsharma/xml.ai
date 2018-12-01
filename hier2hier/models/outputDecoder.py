from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from orderedattrdict import AttrDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OutputDecoder(nn.Module):
    def __init__(self, output_vocab, propagated_info_len, output_decoder_state_width, max_node_count):
        super().__init__()
        self.propagated_info_len = propagated_info_len
        self.output_decoder_state_width = output_decoder_state_width
        self.max_node_count = max_node_count

        self.gruStateLen = propagated_info_len + len(output_vocab)
        self.gruCell = nn.GRU(self.gruStateLen, output_decoder_state_width)

        # Network for symbol decoding.
        self.symbolPreDecoder = nn.Linear(output_decoder_state_width, len(output_vocab))
        self.symbolDecoder = nn.Softmax()
        self.output_vocab = output_vocab

        # Network for attention decoding.
        self.attentionPreDecoder = nn.Linear(output_decoder_state_width + propagated_info_len, max_node_count)
        self.attentionDecoder = nn.Softmax()

    def forward(self, nodeInfoPropagated, nodeAdjacencyInput):
        # Build inputs.
        curSymbol = self.output_vocab.sos_id
        curAttention = Tensor([1 if nodeAdj[0] == i else 0 for i, nodeAdj in enumerate(nodeAdjacencyInput)])

        # Recurrent loop.
        gruCurState = nn.zeros((self.gruStateLen,))
        outputs = []
        while True:
            nodeInfoToAttend = Dot(nodeInfoPropagated, curAttention)
            gruInput = Concat(nodeInfoToAttend, curSymbol)
            gruState = self.GruCell(gruState, gruInput)

            # Compute next symbol.
            curSymbol = self.symbolDecoder(self.symbolPreDecoder(gruState))

            # Compute next attention.
            attentionInputs = Concat(nodeInfosPropagated, gruStat)
            curAttention = self.attentionDecoder(self.attentionPreDecoder(attentionInputs))

            if curSymbol != self.output_vocab.eos_id:
                outputs.append(curSymbol)
            else:
                break
                
        return outputs