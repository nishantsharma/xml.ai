from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from .moduleBase import ModuleBase

class BaseRNN(ModuleBase):
    """
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
    hidden_size (int): number of features in the hidden state `h`
    input_dropout_p (float): dropout probability for the input sequence
    dropout_p (float): dropout probability for the output sequence
    n_layers (int): number of recurrent layers
    rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    
    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self,
            hidden_size,
            input_dropout_p,
            dropout_p,
            n_layers,
            rnn_cell,
            device=None):
        super(BaseRNN, self).__init__(device)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))


    def reset_parameters(self):
        nn.RNN.reset_parameters(self) 

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
