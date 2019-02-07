"""
Module to generate toy1 dataset.
"""

from __future__ import print_function
import argparse
import os
import shutil
import random
import string
import xml.etree.ElementTree as ET
from hier2hier.dataset.randXml import getText

generatorArgsDefaults = {
   "max_len": 10,
}

def addArguments(parser, defaultArgs):
    parser.add_argument('--max-len', help="Max sequence length", default=defaultArgs.max_len)
    return parser

def postProcessArguments(args):
    return args

def generateCommon(appConfig, generatorArgs):
    return None

def generateSample(generatorArgs, commonData):
    """
    Generates input and output XML files for toy1 dataset.
    """
    dataStr =
    retval = ET.Element('toyrev')
    retval.text = getText(1, generatorArgs.max_len))
    return ET.ElementTree(retval)

def transformSample(xmlTree):
    xmlTree.getroot().text = xmlTree.getroot().text[::-1]
    return xmlTree

