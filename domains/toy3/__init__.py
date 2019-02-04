"""
Toy3 dataset is composed of a simple tree with randomly generated children.
The output for each tree is the same tree with following changes.
a) All children reversed.
b) All attributes rotated.
c) All tail and text swapped.

<a><b p1="p1"></b> <c p2="p2"></c></a> becomes <a><c p1="p1"></c> <b p2="p2"><b></a>
"""
from __future__ import print_function
import argparse
import os, copy, shutil, random, string

import xml.etree.ElementTree as ET
from hier2hier.dataset import randomXml, randomVocabs

appConfigDefaults = {
    # AppConfig defaults
    "checkpoint_every": 10,
}

modelArgsDefaults = {
    "attrib_value_vec_len": 32,
    "node_info_propagator_stack_depth": 3,
    "propagated_info_len": 128,
    "output_decoder_stack_depth": 1,
    "output_decoder_state_width": 128,
}
