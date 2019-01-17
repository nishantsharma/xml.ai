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
from .encoderRNN import EncoderRNN
from hier2hier.util import (onehotencode, checkNans, invertPermutation, blockProfiler,
                            methodProfiler, lastCallProfile)


class AttrInfoPropagator(ModuleBase):
    def __init__(self, encoded_attr_len, propagated_info_len, device=None):
        super().__init__(device)

        self.encoded_attr_len = encoded_attr_len
        self.propagated_info_len = propagated_info_len
        self.device = device

        self.layOverAttrInfo = torch.nn.GRUCell(encoded_attr_len, propagated_info_len)
        self.extractAttrInfo = torch.nn.GRUCell(encoded_attr_len, propagated_info_len)
        self.attrOp = nn.Linear(self.encoded_attr_len, self.propagated_info_len)

    def reset_parameters(self, device):
        self.layOverAttrInfo.reset_parameters()
        self.extractAttrInfo.load_state_dict(self.layOverAttrInfo.state_dict())

    @methodProfiler
    def forward(self,
        encodedAttrsByAvdl,
        nodeInfoPropagatedByNdac,
        avdlAttrSelectorsListByNdac,
        decreasingAttrCountsFactor,
        avdl2Ndac,
    ):
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
                accumulatedAttrEncodingByNdac = torch.cat([
                    nodeInfoPropagatedByNdac,
                    nodesWithoutAttrInfoByNdac,
                ])
        else:
            # The case where no node has a child an all are in deficit.
            accumulatedAttrEncodingByNdac = nodeInfoPropagatedByNdac

        return accumulatedAttrEncodingByNdac, attnReadyAttrInfoByAvdl

