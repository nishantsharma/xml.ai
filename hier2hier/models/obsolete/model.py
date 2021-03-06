from __future__ import unicode_literals, print_function, division
from collections import OrderedDict
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from hier2hier.util import blockProfiler, methodProfiler, lastCallProfile, nullTensorBoardHook

from .moduleBase import ModuleBase
from .batchPreProcess import BatchPreProcess
from .nodeInfoEncoder import NodeInfoEncoder
from .nodeInfoPropagator import NodeInfoPropagator
from .outputDecoder import OutputDecoder

curSchemaVersion = 0

class Hier2hier(ModuleBase):
    n = 0
    preprocess_batch = None
    def __init__(self,
            modelArgs,
            debug,
            inputVocabs,
            outputVocab,
            sos_id,
            eos_id,
            device=None):
        super().__init__(device)

        # Split input vocabs.
        (
            tagsVocab,
            textVocab,
            attribsVocab,
            attribValueVocab
        ) = inputVocabs

        # Configure top level items with modelArgs.
        self.max_output_len = modelArgs.max_output_len
        self.outputVocab = outputVocab
        self.debug = debug

        self.nodeInfoEncoder = NodeInfoEncoder(
            tagsVocab,
            textVocab,
            attribsVocab,
            attribValueVocab,
            modelArgs.max_node_count,
            modelArgs.max_node_text_len,
            modelArgs.node_text_vec_len,
            modelArgs.max_attrib_value_len,
            modelArgs.attrib_value_vec_len,
            device=device)

        self.nodeInfoPropagator = NodeInfoPropagator(
            self.nodeInfoEncoder.output_vec_len,
            modelArgs.propagated_info_len,
            modelArgs.max_node_count,
            modelArgs.node_info_propagator_stack_depth,
            modelArgs.disable_batch_norm,
            device=device)

        self.outputDecoder = OutputDecoder(
            outputVocab,
            modelArgs.propagated_info_len,
            modelArgs.output_decoder_state_width,
            modelArgs.output_decoder_stack_depth,
            modelArgs.max_node_count,
            modelArgs.max_output_len,
            sos_id,
            eos_id,
            modelArgs.teacher_forcing_ratio,
            modelArgs.input_dropout_p,
            modelArgs.dropout_p,
            modelArgs.use_attention,
            device=device,
            runtests=self.debug.runtests
            )
            
        if device is not None:
            super().cuda(device)
        else:
            super().cpu()

    def reset_parameters(self):
        self.nodeInfoEncoder.reset_parameters()
        self.nodeInfoPropagator.reset_parameters()
        self.outputDecoder.reset_parameters()

    @methodProfiler
    def forward(self,
            batch_info,
            xmlTreeList,
            targetOutput=None,
            target_lengths=None,
            beam_count=None,
            collectOutput=None,
            tensorBoardHook=None,
            ):
        if tensorBoardHook is None:
            tensorBoardHook = nullTensorBoardHook
        node2Index = {}
        node2Parent = {}
        treeIndex2NodeIndex2NbrIndices = []
        for treeIndex, xmlTree in enumerate(xmlTreeList):
            # Initialize nodeIndex2NbrIndices.
            nodeIndex2NbrIndices = [ ]

            # Build map from node to a unique index for all nodes(including root) in the current tree.
            for nodeIndex, node in enumerate(xmlTree.iter()):
                node2Index[node] = nodeIndex
                nodeIndex2NbrIndices.append(None)

            # Link nodeIndex2NbrIndices with the global treeIndex2NodeIndex2NbrIndices.
            treeIndex2NodeIndex2NbrIndices.append(nodeIndex2NbrIndices)

            # Create an entry for tree root in nodeIndex2NbrIndices.
            # Note: Parent of root is root.
            rootIndex = node2Index[xmlTree.getroot()]
            nodeIndex2NbrIndices[rootIndex] = (rootIndex, [])

            # Create an entry for tree root in node2Parent.
            node2Parent[xmlTree.getroot()] = xmlTree.getroot()

            # Build map from node to its parent for all nodes in current tree.
            for node in xmlTree.iter():
                # The following entry works only because xmlTree iter() is traversing in top down ancestor first manner.
                curNodeChildrenList = nodeIndex2NbrIndices[node2Index[node]][1]
                for childNode in node:
                    node2Parent[childNode] = node
                    nodeIndex2NbrIndices[node2Index[childNode]] = (node2Index[node], [])
                    curNodeChildrenList.append(node2Index[childNode])

        nodeInfoEncodedTensor = self.nodeInfoEncoder(
                node2Index,
                node2Parent,
                xmlTreeList,
                tensorBoardHook,
                )

        nodeInfoPropagatedTensor = self.nodeInfoPropagator(
                treeIndex2NodeIndex2NbrIndices,
                nodeInfoEncodedTensor,
                tensorBoardHook,
                )
        (
            outputSymbolTensors,
            outputSymbols,
            teacherForcedSelections,
            decoderTestTensors
        ) = self.outputDecoder(
                nodeInfoPropagatedTensor,
                targetOutput,
                target_lengths,
                tensorBoardHook,
                beam_count=beam_count,
                collectOutput=collectOutput,
            )

        if self.debug.runtests and targetOutput is not None:
            nodeInfoTensor2 = self.nodeInfoEncoder.test_forward(node2Index, node2Parent, xmlTreeList)
            nodeInfoPropagatedTensor2 = self.nodeInfoPropagator.test_forward(treeIndex2NodeIndex2NbrIndices, nodeInfoEncodedTensor)
            decoderTestTensors2  = self.outputDecoder.test_forward(
                nodeInfoPropagatedTensor,
                targetOutput,
                target_lengths,
                teacherForcedSelections)

            diffSum1 = float(torch.sum(abs(nodeInfoTensor2-nodeInfoEncodedTensor).view(-1)))
            diffSum2 = float(torch.sum(abs(nodeInfoPropagatedTensor2-nodeInfoPropagatedTensor).view(-1)))
            diffSum3 = float(torch.sum(abs(decoderTestTensors2-decoderTestTensors).view(-1)))

            if self.debug.runtests and not self.nodeInfoPropagator.disable_batch_norm:
                print("For properly testing nodeInfoPropagator, disable batch norm by configuring with --disable_batch_norm.")
            print("{0}: diffSums {1:.6f}, {2:.6f}, {3:.6f}".format(
                Hier2hier.n, diffSum1, diffSum2, diffSum3))

            Hier2hier.n += 1

        return outputSymbolTensors, outputSymbols
