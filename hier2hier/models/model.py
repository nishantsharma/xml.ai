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

from .nodeInfoEncoder import NodeInfoEncoder
from .nodeInfoPropagator import NodeInfoPropagator
from .outputDecoder import OutputDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hier2hier(nn.Module):
    def __init__(self,
            modelArgs,
            tagsVocab,
            textVocab,
            attribsVocab,
            attribValueVocab,
            outputVocab):
        super().__init__()
        self.modelArgs = modelArgs
        self.nodeInfoEncoder = NodeInfoEncoder(
            tagsVocab,
            textVocab,
            attribsVocab,
            attribValueVocab,
            modelArgs.max_node_text_len,
            modelArgs.node_text_vec_len,
            modelArgs.max_attrib_value_length,
            modelArgs.attrib_value_vec_len)

        self.nodeInfoPropagator = NodeInfoPropagator(
            self.nodeInfoEncoder.output_vec_len,
            modelArgs.propagated_info_len,
            modelArgs.node_info_propagator_stack_depth)

        outputDecoderConfig = {}
        self.outputDecoder = OutputDecoder(
            outputVocab,
            modelArgs.propagated_info_len,
            modelArgs.output_decoder_state_width,
            modelArgs.max_node_count)

    def forward(self, xmlTreeList, tagetOutput, teacher_forcing_ratio=None):
        nodeAdjacencySpecTensor = torch.zeros((
            len(xmlTreeList),
            self.modelArgs.max_node_count,
            self.modelArgs.max_node_fanout+1))

        for treeIndex, xmlTree in enumerate(xmlTreeList):
            node2Index = {}
            node2parent = { c:p for p in xmlTree.getiterator() for c in p }
            node2parent[xmlTree.getroot()] = xmlTree.getroot()
            # Assign indices to all nodes in current tree.
            for nodeIndex, node in enumerate(xmlTree.iter()):
                node2Index[node] = nodeIndex

            # Build neighborhood tensor.
            for node in xmlTree.iter():
                nodeAdjacencySpecTensor[treeIndex, node2Index[node], 0] = node2Index[node2parent[node]]
                for childIndex, childNode in enumerate(node):
                    nodeAdjacencySpecTensor[treeIndex, node2Index[node], childIndex+1] = node2Index[childNode]

        nodeInfoTensor = self.nodeInfoEncoder(nodeAdjacencySpecTensor, xmlTreeList)
        nodeInfoPropagatedTensor = self.nodeInfoPropagator(nodeAdjacencySpecTensor, nodeInfoTensor)
        treeOutputDecoded = self.outputDecoder(nodeInfoPropagatedTensor)

        return treeOutputDecoded
