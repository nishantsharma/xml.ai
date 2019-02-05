"""
In regular seq2seq, information flows linearly from one sequence position to another.
In hier2hier, information flows linearly within text positions and then across the
XML connectivity graph.

This module implements information flow through attributes.
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from sortedcontainers import SortedDict, SortedSet

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from .moduleBase import ModuleBase
from hier2hier.util import (onehotencode, checkNans, invertPermutation, blockProfiler,
                            methodProfiler, lastCallProfile)


class AttrInfoPropagator(ModuleBase):
    def __init__(self, schemaVersion, encoded_attr_len, propagated_info_len, device=None):
        super().__init__(device, schemaVersion)

        self.encoded_attr_len = encoded_attr_len
        self.propagated_info_len = propagated_info_len
        self.device = device

        self.layOverAttrInfo = torch.nn.GRUCell(encoded_attr_len, propagated_info_len)
        self.extractAttrInfo = torch.nn.GRUCell(encoded_attr_len, propagated_info_len)
        self.attrOp = nn.Linear(self.encoded_attr_len, self.propagated_info_len)

    def reset_parameters(self, device):
        self.layOverAttrInfo.reset_parameters()
        self.extractAttrInfo.load_state_dict(self.layOverAttrInfo.state_dict())

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 0:
            pass
        else:
            super().singleStepSchema(schemaVersion)

    @methodProfiler
    def forward(self,
        encodedAttrsByAvdl,
        nodeInfoPropagatedByNdac,
        avdlAttrSelectorsListByNdac,
        decreasingAttrCountsFactor,
        avdl2Ndac,
    ):
        """
        Implements information flow through attributes.
        Inputs:
            encodedAttrsByAvdl:
                Encoded attributes in AVDL order.
            nodeInfoPropagatedByNdac
                Information propagated upto each XML node, in the NDAC order.
            avdlAttrSelectorsListByNdac
                A list of avdlAttrSelectorsByNdac
                avdlAttrSelectorsByNdac:
                    Each index position ndac within avdlAttrSelectorsByNdac corresponds to
                    an XML node. The value avdl at index position ndac links the XML attribute
                    at index position avdl with XML node at index position ndac.
            decreasingAttrCountsFactor:
                A tensor containing number of attributes of an XML node.
                Provided in decreasing order of values(i.e. in NDAC order).
                Used as Scaling factors.
        Outputs:
            nodeInfoPropagatedByNdac:
                Information(*including attributes info*) propagated upto each XML node,
                in the NDAC order.
            attnReadyAttrInfoByAvdl:
                Atrribute information in the AVDL order, now ready to be attended to.
        """
        inputVecLen = nodeInfoPropagatedByNdac.shape[-1]
        # Create attn ready attr infos, directly from GRU over node state and encoded attrs.
        nodeInfoPropagatedByAvdl = nodeInfoPropagatedByNdac[avdl2Ndac]
        attnReadyAttrInfoByAvdl = self.extractAttrInfo(
            encodedAttrsByAvdl,
            nodeInfoPropagatedByAvdl,
        )

        # Compute children info to propagate to each node.
        encodedAttrValueSumsByNdac = torch.tensor([], device=self.device)

        for avdlAttrSelectorsByNdac in avdlAttrSelectorsListByNdac:
            curEncodedAttrValueByNdac = encodedAttrsByAvdl[avdlAttrSelectorsByNdac, ...]
            if not encodedAttrValueSumsByNdac.shape[0]:
                # First iteration of the loop.
                encodedAttrValueSumsByNdac = curEncodedAttrValueByNdac
            else:
                assert(curEncodedAttrValueByNdac.shape[0] >= encodedAttrValueSumsByNdac.shape[0])
                # If the fanout increases in current iteration, pad neighbor infos by the deficit.
                if curEncodedAttrValueByNdac.shape[0] > encodedAttrValueSumsByNdac.shape[0]:
                    deficit = curEncodedAttrValueByNdac.shape[0] - encodedAttrValueSumsByNdac.shape[0]
                    encodedAttrValueSumsByNdac = nn.ZeroPad2d((0, 0, 0, deficit))(encodedAttrValueSumsByNdac)
                encodedAttrValueSumsByNdac = encodedAttrValueSumsByNdac + curEncodedAttrValueByNdac

        if encodedAttrValueSumsByNdac.shape[0]:
            # Row-wise normalization of childrenInfoToPropagate by fanout.
            # Don't do it in-place.
            decreasingAttrCountsFactor = (
                decreasingAttrCountsFactor
                    .view(list(decreasingAttrCountsFactor.shape) + [1])
                    .expand(encodedAttrValueSumsByNdac.shape)
            )
            accumulatedAttrEncodingByNdac = encodedAttrValueSumsByNdac / decreasingAttrCountsFactor

            # There may still be some row deficit remaining because some nodes do not have children.
            deficitStart = decreasingAttrCountsFactor.shape[0]
            finalDeficit = nodeInfoPropagatedByNdac.shape[0] - deficitStart
            if finalDeficit:
                # Fill blanks using propagatedInfoByAvdl.
                nodesWithoutAttrInfoByNdac = nodeInfoPropagatedByNdac[deficitStart:]

            nodeInfoPropagatedByNdac = self.layOverAttrInfo(
                accumulatedAttrEncodingByNdac,
                nodeInfoPropagatedByNdac[0:deficitStart],
            )

            if finalDeficit:
                nodeInfoPropagatedByNdac = torch.cat([
                    nodeInfoPropagatedByNdac,
                    nodesWithoutAttrInfoByNdac,
                ])

        assert(inputVecLen == nodeInfoPropagatedByNdac.shape[-1])

        return nodeInfoPropagatedByNdac, attnReadyAttrInfoByAvdl

