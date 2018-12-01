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

class TagEncoder(nn.Module):
    def __init__(self, tagsVocab):
        super().__init__()
        self.tagsVocab = tagsVocab

    def __onehotencode(self, i):
        return [1 if j==i else 0 for j in range(len(self.tagsVocab))]

    def forward(self, tagLists):
        return torch.Tensor([self.onhotencode(self.tagsVocab[tag]) for tag in tagList])

    @property
    def output_vec_len(self):
        return len(self.tagsVocab)

class AttribsEncoder(nn.Module):
    def __init__(self, attribsVocab, attribValueEncoder):
        super().__init__()
        self.attribsVocab = attribsVocab
        self.attribValueEncoder = attribValueEncoder

    def __onehotencode(self, i):
        return [1 if j==i else 0 for j in range(len(self.tagsVocab))]

    @property
    def output_vec_len(self):
        return len(self.attribsVocab) * self.attribValueEncoder.output_vec_len[1]

    def forward(self, attribsDictsList):
        sampleCount = len(attribDictsList)
        attrCount = len(self.attribsVocab)
        attrVecLen = len(self.attribValueEncoder(""))
        retval = torch.zeros([sampleCount, attrCount, attrVecLen])
        for sampleIndex, attribsDict in enumerate(attribsDictsList):
            for attribName, atttribValue in attribsDict.items():
                retval[sampleIndex][self.attribsVocab[attribName]] = self.attribValueEncoder(attribValue)
        return torch.Tensor([self.__onhotencode(self.tagsVocab[tag]) for tag in tags])

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
            max_node_text_len,
            node_text_vec_len,
            max_attrib_value_length,
            attrib_value_vec_len):
        super().__init__()

        # Build component encoders.
        self.tagEncoder = TagEncoder(tagsVocab)
        self.nodeTextEncoder = EncoderRNN(len(textVocab), max_node_text_len, len(textVocab), node_text_vec_len)
        self.attribValueEncoder = EncoderRNN(len(attribValueVocab), max_attrib_value_length, len(attribValueVocab), attrib_value_vec_len)
        self.attribsEncoder = AttribsEncoder(attribsVocab, self.attribValueEncoder)

    @property
    def output_vec_len(self):
        retval = 0
        retval += self.tagEncoder.output_vec_len
        retval += self.nodeTextEncoder.output_vec_len[1]
        retval += self.attribsEncoder.output_vec_len
        return retval

    def forward(self, nbrhoodSpecTensor, xmlTreeList):
        encodedTags = self.labelsEncoder(xmlTreeList)
        encodedText = self.nodeTextEncoder(xmlTreeList)
        encodedAttributes = self.labelsEncoder(xmlTreeList)

        retval = Concat(encodedTags, encodedText, encodedAttributes)
        return retval
