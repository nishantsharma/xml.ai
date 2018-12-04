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

from .encoderRNN import EncoderRNN

def onehotencode(n, i):
    return [1 if j==i else 0 for j in range(n)]

class TagEncoder(nn.Module):
    def __init__(self, tagsVocab, max_node_count):
        super().__init__()
        self.tagsVocab = tagsVocab
        self.max_node_count = max_node_count

    def __onehotencode(self, i):
        return [1 if j==i else 0 for j in range(len(self.tagsVocab))]

    def forward(self, node2Index, node2Parent, xmlTreeList):
        retval = torch.zeros((len(xmlTreeList), self.max_node_count, len(self.tagsVocab)))
        for index, xmlTree in enumerate(xmlTreeList):
            for node in xmlTree.iter():
                retval[index, node2Index[node]] = torch.Tensor(onehotencode(len(self.tagsVocab), self.tagsVocab.stoi[node.tag]))

        return retval

    @property
    def output_vec_len(self):
        return len(self.tagsVocab)

class NodeTextEncoder(EncoderRNN):
    def __init__(self, textVocab, max_node_count, max_node_text_len, node_text_vec_len):
        super().__init__(
            len(textVocab),
            max_node_text_len,
            len(textVocab),
            node_text_vec_len,
            variable_lengths=True,
            bidirectional=False)
        self.textVocab = textVocab
        self.max_node_count = max_node_count
        self.node_text_vec_len = node_text_vec_len

    def forward(self, node2Index, node2Parent, xmlTreeList):
        maxLen = -1
        allText = []
        for treeIndex, xmlTree in enumerate(xmlTreeList):
            for nodeIndex, node in enumerate(xmlTree.iter()):
                allText.append((treeIndex, nodeIndex, node.text))

        # Sort all text according to length, largest first.
        allText.sort(key=lambda x: - len(x[2]))

        # Create inputs appropriate for use by EncoderRNN.
        maxTextLen = len(allText[0][2])
        textTensor = torch.zeros([len(allText), maxTextLen], dtype=torch.long)
        textLengthsTensor = torch.zeros(len(allText), dtype=torch.int32)
        for textIndex, (_, _, text) in enumerate(allText):
            for chIndex, ch in enumerate(text):
                textTensor[textIndex, chIndex] = self.textVocab.stoi[ch]
            textLengthsTensor[textIndex] = len(text)

        # Encode.
        _, encodedTextVec = super().forward(textTensor, textLengthsTensor)

        # Populate retval
        retval = torch.zeros((len(xmlTreeList), self.max_node_count, self.node_text_vec_len))
        for encodedIndex, (treeIndex, nodeIndex, _) in enumerate(allText):
            retval[treeIndex, nodeIndex] = encodedTextVec[0][encodedIndex]

        return retval

class AttribsEncoder(nn.Module):
    def __init__(self, attribsVocab, attribValueEncoder, max_node_count):
        super().__init__()
        self.attribsVocab = attribsVocab
        self.attribValueEncoder = attribValueEncoder
        self.max_node_count = max_node_count

    @property
    def output_vec_len(self):
        return len(self.attribsVocab) * self.attribValueEncoder.output_vec_len[1]

    def forward(self, node2Index, node2Parent, xmlTreeList):
        sampleCount = len(xmlTreeList)
        attrCount = len(self.attribsVocab)
        attrVecLen = self.attribValueEncoder.hidden_size
        retval = torch.zeros([sampleCount, self.max_node_count, attrCount, attrVecLen])
        for sampleIndex, xmlTree in enumerate(xmlTreeList):
            for node in xmlTree.iter():
                for attribName, atttribValue in node.attrib.items():
                    attrbVec = self.attribValueEncoder(attribValue)
                    attribIndex = self.attribsVocab.stoi[attribName]
                    retval[sampleIndex, node2Index[node], attribIndex] = attribVec
        
        return retval

class NodeInfoEncoder(nn.Module):
    """
    Applies a multi layer GRU to an input character sequence.

    rnn_cell (str, optional): type of RNN cell (default: gru)
    """
    def __init__(self,
            tagsVocab,
            textVocab,
            attribsVocab,
            attribValueVocab,
            max_node_count,
            max_node_text_len,
            node_text_vec_len,
            max_attrib_value_length,
            attrib_value_vec_len):
        super().__init__()

        # Build component encoders.
        self.tagsEncoder = TagEncoder(tagsVocab, max_node_count)
        self.nodeTextEncoder = NodeTextEncoder(textVocab, max_node_count, max_node_text_len, node_text_vec_len)
        self.attribValueEncoder = EncoderRNN(len(attribValueVocab), max_attrib_value_length, len(attribValueVocab), attrib_value_vec_len)
        self.attribsEncoder = AttribsEncoder(attribsVocab, self.attribValueEncoder, max_node_count)

    @property
    def output_vec_len(self):
        retval = 0
        retval += self.tagsEncoder.output_vec_len
        retval += self.nodeTextEncoder.output_vec_len[1]
        retval += self.attribsEncoder.output_vec_len
        return retval

    def forward(self, node2Index, node2Parent, xmlTreeList):
        encodedTags = self.tagsEncoder(node2Index, node2Parent, xmlTreeList)
        encodedText = self.nodeTextEncoder(node2Index, node2Parent, xmlTreeList)
        encodedAttributes = self.attribsEncoder(node2Index, node2Parent, xmlTreeList)

        attrShape = encodedAttributes.shape
        newAttrShape = attrShape[0:-2] + (attrShape[-1] * attrShape[2],)
        encodedAttributesReshaped = encodedAttributes.reshape(newAttrShape)

        return torch.cat([encodedTags, encodedText, encodedAttributesReshaped], -1)
