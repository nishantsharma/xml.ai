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

from nodeInfoEncoder import NodeInfoEncoder
from nodeInfoPropagator import NodeInfoPropagator
from nodeDecoder import NodeDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class xmlLearner(nn.Module):
    def __init__(self):
        self.nodeInfoEncoder = NodeInfoEncoder()
        self.nodeInfoPropagator = NodeInfoPropagator()
        self.nodeDecoder = NodeDecoder()

    def forward(self, xmlTree):
        nodeInfoCodes = self.nodeInfoEncoder(xmlTree)
        nodeInfoCodesPropagated = self.nodeInfoPropagator(nodeInfosCode)
        treeOutputDecoded = self.nodeDecoder(nodeInfoCodesPropagated)

        return treeOutputDecoded