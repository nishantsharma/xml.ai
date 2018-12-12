from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from orderedattrdict import AttrDict
from sortedcontainers import SortedDict, SortedSet

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from .moduleBase import ModuleBase 
from .encoderRNN import EncoderRNN
from .utils import onehotencode, checkNans

class TagEncoder(ModuleBase):
    def __init__(self, tagsVocab, max_node_count, device=None):
        super().__init__(device)
        self.tagsVocab = tagsVocab
        self.max_node_count = max_node_count

    def forward(self, node2Index, node2Parent, xmlTreeList):
        allTreeCodes = []
        for index, xmlTree in enumerate(xmlTreeList):
            treeCode = []
            for index, node in enumerate(xmlTree.iter()):
                treeCode.append(onehotencode(len(self.tagsVocab), self.tagsVocab.stoi[node.tag]))
                assert(index == node2Index[node])
            allTreeCodes.append(treeCode)

        return torch.tensor(allTreeCodes, device=self.device)

    @property
    def output_vec_len(self):
        return len(self.tagsVocab)

class NodeTextEncoder(EncoderRNN):
    def __init__(self, textVocab, max_node_count, max_node_text_len, node_text_vec_len, device=None):
        super().__init__(
            len(textVocab),
            max_node_text_len,
            len(textVocab),
            node_text_vec_len,
            variable_lengths=True,
            bidirectional=False,
            device=device)
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
        allTextLengthSorted = sorted(allText, key=lambda x: - len(x[2]))

        # Create inputs appropriate for use by EncoderRNN.
        maxTextLen = len(allTextLengthSorted[0][2])
        textList = []
        textLengthsList = []
        for textIndex, (_, _, text) in enumerate(allTextLengthSorted):
            # Expand current text as a list.
            curTextAsList = [self.textVocab.stoi[ch] for ch in text]

            # Record length of the current text.
            textLengthsList.append(len(curTextAsList))

            # Pad curTextAsList and add to textList.
            curTextAsListPadded = curTextAsList + [self.textVocab.stoi["<pad>"]] * (maxTextLen - len(text))
            textList.append(curTextAsListPadded)
        textTensor = torch.tensor(textList, dtype=torch.long, device=self.device)
        textLengthsTensor = torch.tensor(textLengthsList, dtype=torch.long, device=self.device)

        # Encode.
        _, encodedTextVec = super().forward(textTensor, textLengthsTensor)

        # Populate treeIndex2NodeIndex2EncodedIndex
        treeIndex2NodeIndex2EncodedIndex = SortedDict()
        for encodedIndex, (treeIndex, nodeIndex, _) in enumerate(allTextLengthSorted):
            nodeIndex2EncodedIndex = treeIndex2NodeIndex2EncodedIndex.setdefault(
                treeIndex,
                SortedDict())
            nodeIndex2EncodedIndex[nodeIndex] = encodedIndex

        # Get back node text indices in original order.
        allTextEncoded = []
        zeroPaddingNodeVec = torch.zeros([self.node_text_vec_len], device=self.device)
        for treeIndex, nodeIndex2EncodedIndex in treeIndex2NodeIndex2EncodedIndex.items():
            treeTextEncoded = [
                encodedTextVec[:, encodedIndex, :]
                for nodeIndex, encodedIndex in nodeIndex2EncodedIndex.items()
            ]

            # Some of the trees may need padding so tht all trees have self.max_node_count
            # number of node vectors.
            treeTextEncoded += [zeroPaddingNodeVec] * (self.max_node_count - len(treeTextEncoded))
            treeTextEncoded = torch.cat(treeTextEncoded).view(1, self.max_node_count, self.node_text_vec_len)
            allTextEncoded.append(treeTextEncoded)
        allTextEncoded = torch.cat(allTextEncoded)

        return allTextEncoded

class AttribsEncoder(ModuleBase):
    def __init__(self, attribsVocab, attribValueEncoder, max_node_count, device=None):
        super().__init__(device)
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
        retval = []
        #torch.zeros([sampleCount, self.max_node_count, attrCount, attrVecLen], device=self.device)
        zeroAttrib = torch.zeros(1, attrVecLen, device=self.device)
        for treeIndex, xmlTree in enumerate(xmlTreeList):
            encodedTree = []
            for node in xmlTree.iter():
                encodedNode = [zeroAttrib for _ in range(attrCount)]
                for attribName, atttribValue in node.attrib.items():
                    attribVec = self.attribValueEncoder(attribValue)
                    attribIndex = self.attribsVocab.stoi[attribName]
                    encodedNode[attribIndex] = attribVec
                    checkNans(attribVec)
                encodedNode = torch.cat(encodedNode).view(1, attrCount, attrVecLen)
                encodedTree.append(encodedNode)
            encodedTree = torch.cat(encodedTree).view(1, self.max_node_count, attrCount, attrVecLen)
            retval.append(encodedTree)
            checkNans(encodedTree)

        retval = torch.cat(retval)
        return retval

class NodeInfoEncoder(ModuleBase):
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
            max_attrib_value_len,
            attrib_value_vec_len,
            device=None):
        super().__init__(device)

        # Build component encoders.
        self.tagsEncoder = TagEncoder(tagsVocab, max_node_count, device=device)
        self.nodeTextEncoder = NodeTextEncoder(
                textVocab,
                max_node_count,
                max_node_text_len,
                node_text_vec_len,
                device=device)
        self.attribValueEncoder = EncoderRNN(
                len(attribValueVocab),
                max_attrib_value_len,
                len(attribValueVocab),
                attrib_value_vec_len,
                device=device)
        self.attribsEncoder = AttribsEncoder(
                attribsVocab,
                self.attribValueEncoder,
                max_node_count,
                device=device)

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
        checkNans(encodedTags)
        checkNans(encodedText)
        checkNans(encodedAttributes)

        attrShape = encodedAttributes.shape
        newAttrShape = attrShape[0:-2] + (attrShape[-1] * attrShape[-2],)
        encodedAttributesReshaped = encodedAttributes.view(newAttrShape)

        retval = torch.cat([encodedTags, encodedText, encodedAttributesReshaped], -1)
        checkNans(retval)
        return retval
