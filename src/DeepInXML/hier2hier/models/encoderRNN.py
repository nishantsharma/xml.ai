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

from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    """
    Applies a multi layer RNN to an input character sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of       token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden      state `h`

    Examples::

        >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
        >>> output, hidden = encoder(input)

    """
    def __init__(self, vocab_size, max_len, input_size, hidden_size,
            input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell="gru", variable_lengths=False,
            doEmbed=True, update_embedding=False, device=None):
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell, device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.variable_lengths = variable_lengths
        self.vocab = torch.eye(vocab_size, device=device)
        if doEmbed:
            self.embedding = nn.Embedding(vocab_size, input_size)
            self.embedding.weight = nn.Parameter(self.vocab)
            self.embedding.weight.requires_grad = update_embedding
            self.rnn_input_size = input_size
        else:
            self.embedding = None

            self.rnn_input_size = vocab_size
        self.rnn = self.rnn_cell(self.rnn_input_size, hidden_size, n_layers,
                                batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    @property
    def output_vec_len(self):
        return self.input_size, self.hidden_size

    def forward(self, input_var, input_lengths=None):
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
        # Embed the input(s).
        embedded = self.vocab[input_var] if self.embedding is None else self.embedding(input_var)
        
        # Dropout some of the inputs, when configured.
        embedded = self.input_dropout(embedded)

        # If the inputs are variable lengths, we need to pack the sequences.
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 

        # Run the RNN.
        output, hidden = self.rnn(embedded)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
