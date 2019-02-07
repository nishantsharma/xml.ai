"""
A simple wrapper around torch RNN modules.
"""
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
import torch.nn.utils.rnn as rnn
from .moduleBase import ModuleBase
from hier2hier.util import blockProfiler, methodProfiler

class EncoderRNN(ModuleBase):
    """
    Applies a multi layer RNN to an input character sequence.

    Args:
        rnn_input_size (int): size of the RNN input.
        hidden_size (int): the number of features in the hidden state `h`
        doEmbed (bool): Set to true, if inputs are ordinal values and RNN input is an encoded
                        version of it.
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell. Options: GRU/LSTM. Default: GRU.
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of
            token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden
            state `h`

    Examples::

        >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
        >>> output, hidden = encoder(input)

    """
    def __init__(self, schemaVersion, rnn_input_size, hidden_size, doEmbed=True,
            input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False,
            rnn_cell="gru",
            update_embedding=True, vocab_size=None, initialEmbedding=None, device=None):
        super().__init__(schemaVersion, device)
        
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size

        if vocab_size is None:
            vocab_size = rnn_input_size

        # The input embedding.
        if doEmbed:
            if initialEmbedding is None:
                initialEmbedding = torch.eye(rnn_input_size, device=device)
            self.embedding = nn.Embedding(vocab_size, rnn_input_size)
            self.embedding.weight = nn.Parameter(initialEmbedding)
            self.embedding.weight.requires_grad = update_embedding
        else:
            assert(vocab_size == rnn_input_size)
            self.vocab = torch.eye(vocab_size, device=device)
            self.embedding = None

        # Dropout module for regularization.
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        # RNN cell, the work horse.
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
            
        self.rnn = self.rnn_cell(rnn_input_size, hidden_size, n_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=dropout_p)

    def reset_parameters(self, device):
        if self.embedding is not None:
            self.embedding.reset_parameters()

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 0:
            pass
        else:
            super().singleStepSchema(schemaVersion)

    @property
    def output_vec_len(self):
        return self.input_size, self.hidden_size

    @methodProfiler
    def forward(self, input_var, initial_hidden=None, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len):
                tensor containing features of the input sequence.
            initial_hidden (num_layers * num_directions, batch, hidden_size)
            input_lengths (list of int, optional):
                A list that contains the lengths of sequences in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the       hidden state h
        """
        # Embed the "discrete" input(s).
        BatchSizesDiscovered = None
        if self.embedding is None:
            embedded = self.vocab[input_var]
        else:
            if isinstance(input_var, rnn.PackedSequence):
                embedded = self.embedding(input_var.data)
                BatchSizesDiscovered = input_var.batch_sizes
            else:
                embedded = self.embedding(input_var)
        
        # Dropout some of the inputs, when configured.
        if embedded.shape[0]:
            embedded = self.input_dropout(embedded)

        # If the inputs are variable lengths, we need to pack the sequences.
        if BatchSizesDiscovered is not None:
            embedded = rnn.PackedSequence(embedded, BatchSizesDiscovered) 
        elif input_lengths is not None:
            embedded = rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 

        # Run the RNN.
        if initial_hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            if not isinstance(embedded, rnn.PackedSequence):
                if embedded.shape[0]:
                    output, hidden = self.rnn(embedded, initial_hidden)
                else:
                    output, hidden = embedded.view(1, 0, self.hidden_size), initial_hidden
            else:
                # initial_hidden(call it initial_state) may be bigger in
                # batch dimension than embedded(call it inputs).
                # That just means that we shortcut uncovered portion of initial_hidden
                # to the output.
                noInputsStart, noInputsEnd = int(embedded.batch_sizes[0]), initial_hidden.shape[1]
                output, hidden = self.rnn(embedded, initial_hidden[:,0:noInputsStart,...])
                if noInputsStart != noInputsEnd:
                    hidden = torch.cat([hidden, initial_hidden[:,noInputsStart:,...]], dim=1)

        if BatchSizesDiscovered is None and input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        hidden = hidden.view(hidden.shape[1], self.hidden_size)
        return output, hidden
