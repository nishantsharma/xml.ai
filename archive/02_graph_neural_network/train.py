"""
This model figures out transformations in hiearchical data.
"""

from __future__ import print_function

import json, argparse
import numpy as np
from orderedattrdict import AttrDict
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model
from utils import saveModel
from inputs import get_inputs
from nodeAttrsEncoder import NodeAttrsEncoder
from nodeDecoder import NodeDecoder


# Build training args needed during training and also inference.
cmdParser = argparse.ArgumentParser
cmdParser.add_argument("--batch_size", type = int, default = 64,
                help="Batch size for training.")
cmdParser.add_argument("--epochs", type = int, default = 100,
                help="Number of epochs to train for.")
cmdParser.add_argument("--num_samples", type = int, default = 10000,
                help="Number of samples to train on.")

# Schema params.
cmdParser.add_argument("--total_attrs_count", type = int, default = 128,
                help="Total number of known attributes in the schema..")
cmdParser.add_argument("--value_symbols_count", type = int, default = 256,
                help="Total number of symbols used in attribute value strings..")
cmdParser.add_argument("--max_node_count", type = int, default = 8192,
                help="Maximum number of nodes in an XML file..")
cmdParser.add_argument("--max_node_name_len", type = int, default = 256,
                help="Maximum length of name of any node.")
cmdParser.add_argument("--max_node_fanout", type = int, default = 128,
                help="Maximum connectivity fanout of an XML node.")
cmdParser.add_argument("--max_node_attr_len", type = int, default = 256,
                help="Maximum length of any attribute value at any node.")
cmdParser.add_argument("--max_xmlfile_len", type = int, default = 16536,
                help="Maximum length of an input XML file.")
cmdParser.add_argument("--max_output_len", type = int, default = 16536,
                help="Maximum length of the output file.")

# Size meta-parameters of the generated neural network.
cmdParser.add_argument("--graph_encoder_stack_depth", type = int, default = 12,
                help="Depth of the graph layer stack. This determines the number of "
                + "hops that information would travel in the graph, during node encoding.")
cmdParser.add_argument("--node_info_vec_len", type = int, default = 64,
                help="Length of node_info vector.")
cmdParser.add_argument("--node_decoder_stack_depth", type = int, default = 6,
                help="Stack depth of node decoder.")
cmdParser.add_argument("--node_decoder_gru_width", type = int, default = 6,
                help="Width of GRU cell in node decoder.")


# Parse the arguments and build the dictionary.
modelArgs = cmdParser.parse_args()

# NodeAttrsEncoder encodes attribuets of tree nodes.
nodeAttrsEncoder = NodeAttrsEncoder(modelArgs)

# Run the filter and get output data.
nodeNamesTensor, nodeAttrsTensor, nodeAdjacencySpecTensor = nodeAttrsEncoder.inputs()
nodeInfoCodeTensor, = nodeAttrsEncoder.outputs()

# Tree info decoder uses node info code and node associated attention to decode the entire tree.
nodeDecoder = NodeDecoder(modelArgs, nodeInfoCodeTensor)

# Run the transformer and get output data.
lastDecodedValueTensor = nodeDecoder.inputs()
decodedValueTensor, = nodeDecoder.outputs()

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
trainer_model = Model(
        [ nodeNamesTensor, nodeAttrsTensor, nodeAdjacencySpecTensor ],
        [ decodedValueTensor ])

# Now, get training inputs.
inputData = get_inputs(modelArgs)

plot_model(trainer_model, to_file="plot.png")

# Run training
trainer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
trainer_model.fit(
        [
            inputData.node_names_data,
            inputData.node_attrs_data,
            inputData.node_adjacency_spec_data,
            inputData.node_attention_data],
        inputData.decoder_target_data,
        batch_size=modelArgs.batch_size,
        epochs=modelArgs.epochs,
        validation_split=0.2)

# Save model
saveModel(trainer_model, 't_s2s.json')
with open("modelArgs.json", "w") as fp:
    json.dump(modelArgs, fp)

