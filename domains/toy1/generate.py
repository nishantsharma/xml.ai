"""
Module to generate toy1 dataset.
"""

from __future__ import print_function
import argparse
import os
import shutil
import random
import string
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

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

srcAlphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
def generateSample(rootFolder, datasetName, treeCount):
    """
    Generates input and output XML files for toy1 dataset.
    """
    length = random.randint(1, args.max_len)
    dataSeq = []
    dataStr = ""
    for _ in range(length):
        ch = srcAlphabet[random.randint(0, len(srcAlphabet)-1)]
        dataSeq.append(str(ord(ch)))
        dataStr += ch

    retval = ET.Element('toyrev')
    retval.text = dataStr
    return retval

def transformSample(xmlTree):
    xmlTree.text = xmlTree.text[::-1]
    return xmlTree

