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

import json, argparse
import numpy as np
from orderedattrdict import AttrDict
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model
from utils import saveModel
from inputs import get_inputs

# Build training args needed during training and also inference.
cmdParser = argparse.ArgumentParser
cmdParser.add_argument("--batch_size", type = int, default = 64,
        help="Batch size for training.")
cmdParser.add_argument("--epochs", type = int, default = 100,
        help="Number of epochs to train for.")
cmdParser.add_argument("--num_samples", type = int, default = 10000,
        help="Number of samples to train on.")

# Model Hyper parameters
# .
cmdParser.add_argument("--attrs_code_dim", type = int, default = 256,
        help="Size of vector space used to encode all attributes of a node.")
cmdParser.add_argument("--children_code_dim", type = int, default = 512,
        help="Size of vector space used to encode all children of a node. Encodes mostly pointers.")
cmdParser.add_argument("--attrs_decoder_dim", type = int, default = 512,
        help="Latent dimensionality of attributes encoding space.")
cmdParser.add_argument("--children_decoder_dim", type = int, default = 512,
        help="Latent dimensionality of attributes encoding space.")
cmdParser.add_argument("--max_node_attributes_count", type = int, default = 32,
        help="Maximum number of attributes in a node.")
cmdParser.add_argument("--total_schema_attributes_count", type = int, default = 128,
        help="Maximum number of attributes in a single node.")
cmdParser.add_argument("--total_schema_attributes_count", type = int, default = 128,
        help="Maximum number of attributes in the entire schema.")
cmdParser.add_argument("--max_attribute_values_len", type = int, default = 128,
        help="Maximum length of attributes values.")
cmdParser.add_argument("--max_node_children_count", type = int, default = 128,
        help="Maximum number of children of a node.")


# Parse the arguments and build the dictionary.
modelArgs = cmdParser.parse_args()

# Derived model params.
# Dimensionality of tensor representing complete transferable information that flows up to the parent node.
modelArgs.node_code_dim = modelArgs.children_code_dim + modelArgs.attrs_code_dim

# Path to the data txt file on disk.
modelArgs.data_path = 'fra-eng/fra.txt'

# Instantiate codecs.
attributesCodec = AttributesCodec()
childrenCodec = ChildrenCodec()

# Encode attributes.
attributesEncoderInputs = attributesCodec.encoderInputs()
attributesEncoderOutputs = attributesCodec.encoder(attrEncoderInputs)

# Encode children.
childrenEncoderInputs = childrenCodec.encoderInputs()
childrenEncoderOutputs = childrenCodec.encoder(childrenEncoderInputs)

# Decode new attributes
newAttrbutesTensor = attribuetsCodec.decode(attributesEncoderOutputs, childrenEncoderOutputs)

# Decode children
newChildrenTensor = childrenCodec.decode(attributesEncoderOutputs, childrenEncoderOutputs)

decoderInputs = [attrCodeTensor, childrenCodeTensor]

# Apply decoder on encoder outputs.
childrenCodeActivated = childrenCodec.decode(decoderInputs)
attrsCodeActivated = attributeCodec.decode(decoderInputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
trainer_model = Model(
        [*attributesEncoderInputs, *childrenEncoderInputs, *decoderInputs],
        [childrenCodeActivated, attrsCodeActivated])

# Get inputs.
inputData = get_inputs(modelArgs)

plot_model(trainer_model, to_file="plot.png")

# Run training
trainer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
trainer_model.fit(
        [inputData.encoder_input_data, inputData.decoder_input_data],
        inputData.decoder_target_data,
        batch_size=modelArgs.batch_size,
        epochs=modelArgs.epochs,
        validation_split=0.2)

# Save model
saveModel(trainer_model, 't_s2s.json')
with open("modelArgs.json", "w") as fp:
    json.dump(modelArgs, fp)

