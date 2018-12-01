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

from encoderRNN import EncoderRNN

class NodeInfoEncoder(nn.Module):
    """
    Applies a multi layer GRU to an input character sequence.

    rnn_cell (str, optional): type of RNN cell (default: gru)
    """
    def __init__(self, labelsEncoderConfig, attrbutesEncoderConfig):
        super().__init__()
        self.labelsEncoder = EncoderRNN(
            len(labelsEncoderConfig.vocab),
            labelsEncoderConfig.max_lebel_length,
            labelsEncoderConfig.hidden_size,
            variable_lengths=True,
        )

        self.attributesEncoderConfig = attrbutesEncoderConfig

    def forward(self, names, nameLengths, attributeValues, attributeValueLengths):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the       hidden state h
        """
        encodedNames = self.labelsEncoder(names, nameLengths)
        encodedAttributes = self.labelsEncoder(attributeValues, attributeValueLengths)

        return (encodedNames, encodedAttributes)
