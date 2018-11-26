#!python
'''Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

import json
import numpy as np
from attrdict import AttrDict
from keras.models import Model
from keras.layers import Input, LSTM, Dense

from utils import saveModel
from inputs import get_inputs 

# Build training args needed during training and also inference.
trainArgs = AttrDict()
trainArgs.batch_size = 64  # Batch size for training.
trainArgs.epochs = 100  # Number of epochs to train for.
trainArgs.latent_dim = 256  # Latent dimensionality of the encoding space.
trainArgs.num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
trainArgs.data_path = 'fra-eng/fra.txt'

# Get inputs.
inputs = get_inputs(trainArgs)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, trainArgs.num_encoder_tokens))
encoder = LSTM(trainArgs.latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, trainArgs.num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(trainArgs.latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(trainArgs.num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
trainer_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
trainer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
trainer_model.fit(
        [inputs.encoder_input_data, inputs.decoder_input_data],
        inputs.decoder_target_data,
        batch_size=trainArgs.batch_size,
        epochs=trainArgs.epochs,
        validation_split=0.2)

# Save model
saveModel(trainer_model, 't_s2s.json')
with open("trainArgs.json", "w") as fp:
    json.dump(trainArgs, fp)

