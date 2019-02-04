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
import os, copy, shutil, random, glob, string

import xml.etree.ElementTree as ET
from hier2hier.dataset import randomXml, randomVocabs
from hier2hier.util import str2bool

generatorArgsDefaults = {
    # Node generation.
    "node_count_min": 2,
    "node_count_max": 5,
    "max_child_count": 3,

    # Attr generation
    "attr_count_min": 0,
    "attr_count_max": 2,

    # Text generation
    "text_len_min": -10,
    "text_len_max": 3,

    # Tail generation
    "tail_len_min": -20,
    "tail_len_max": 3,

    # Tag selection pool.
    "tag_pool_count": 30,
    "tag_len_min": 1,
    "tag_len_max": 3,

    # Attr Label selection pool.
    "attr_pool_count": 20,
    "attr_len_min": 1,
    "attr_len_max": 3,

    # Attr Value selection pool.
    "attr_value_pool_count": 30,
    "attr_value_len_min": 1,
    "attr_value_len_max": 3,

    # Load existing vocab
    "load_existing_vocab": True,
}

def addArguments(parser, defaultArgs):
    # Load existing vocab.
    parser.add_argument("--load_existing_vocab", default=defaultArgs.load_existing_vocab,
            type = str2bool,
            help="If set, we reuse existing set of node tags, attr labels and attr values.")

    # Node generation.
    parser.add_argument("--node_count_min", default=defaultArgs.node_count_min,
            help="Minimum number of nodes in XML tree.")
    parser.add_argument("--node_count_max", default=defaultArgs.node_count_max,
            help="Maximum number of nodes in XML tree.")
    parser.add_argument("--max_child_count", default=defaultArgs.max_child_count,
            help="Maximum fanout of a node in XML tree.")

    # Attr generation
    parser.add_argument("--attr_count_min", default=defaultArgs.attr_count_min,
            help="Minimum number of attrs of a nodes in XML tree.")
    parser.add_argument("--attr_count_max", default=defaultArgs.attr_count_max,
            help="Maximum number of attrs of a nodes in XML tree.")

    # Text generation
    parser.add_argument("--text_len_min", default=defaultArgs.text_len_min,
            help="Minimum length of text field of a node in XML tree.")
    parser.add_argument("--text_len_max", default=defaultArgs.text_len_max,
            help="Maximum length of text field of a node in XML tree.")

    # Tail generation
    parser.add_argument("--tail_len_min", default=defaultArgs.tail_len_min,
            help="Minimum length of tail field of a node in XML tree.")
    parser.add_argument("--tail_len_max", default=defaultArgs.tail_len_max,
            help="Maximum length of tail field of a node in XML tree.")

    # Tag selection pool.
    parser.add_argument("--tag_pool_count", default=defaultArgs.tag_pool_count,
            help="Maximum tags in the tag pool to pick from.")
    parser.add_argument("--tag_len_min", default=defaultArgs.tag_len_min,
            help="Minimum length of tag ID of a node in XML tree.")
    parser.add_argument("--tag_len_max", default=defaultArgs.tag_len_max,
            help="Maximum length of tag ID of a node in XML tree.")

    # Attr Label selection pool.
    parser.add_argument("--attr_pool_count", default=defaultArgs.attr_pool_count,
            help="Maximum attr labels in the attr label pool to pick from.")
    parser.add_argument("--attr_len_min", default=defaultArgs.attr_len_min,
            help="Minimum length of label of an attr in XML tree.")
    parser.add_argument("--attr_len_max", default=defaultArgs.attr_len_max,
            help="Maximum length of label of an attr in XML tree.")

    # Attr Value selection pool.
    parser.add_argument("--attr_value_pool_count", default=defaultArgs.attr_value_pool_count,
            help="Maximum values in the attr value pool to pick from.")
    parser.add_argument("--attr_value_len_min", default=defaultArgs.attr_value_len_min,
            help="Minimum length of value of an attr in XML tree.")
    parser.add_argument("--attr_value_len_max", default=defaultArgs.attr_value_len_max,
            help="Maximum length of value of an attr in XML tree.")

    return parser

def postProcessArguments(args):
    return args

def generateCommon(appConfig, generatorArgs):
    ga = generatorArgs
    if ga.load_existing_vocab:
        nodeTags = set()
        nodeAttrLabels = set()
        nodeAttrValues = set()
        existingPath=appConfig.inputs_root_dir + appConfig.domain + "/"
        allFiles = glob.glob(existingPath + "/*/*.xml")
        for filePath in allFiles:
            for xmlNode in ET.parse(filePath).iter(): 
                nodeTags.add(xmlNode.tag)
                for attrLabel, attrValue in xmlNode.attrib.items():
                    nodeAttrLabels.add(attrLabel)
                    nodeAttrValues.add(attrValue)

        return list(nodeTags), list(nodeAttrLabels), list(nodeAttrValues)  
    else:
        vocabArgs = {
            # (tag Count, (min tag length, max tag length))
            "tag_gen_params": (ga.tag_pool_count, (ga.tag_len_min, ga.tag_len_max)),

            # (attr Count, (min attr length, max attr length))
            "attr_gen_params": (ga.attr_pool_count, (ga.attr_len_min, ga.attr_len_max)),

            # (attr value Count, (min attr value length, max attr value length))
            "attr_value_gen_params": (ga.attr_value_pool_count, (ga.attr_value_len_min, ga.attr_value_len_max)),
        }
        return randomVocabs(vocabArgs)

def generateSample(generatorArgs, vocabs):
    ga = generatorArgs
    treeArgs = {
        "node_count_range": (ga.node_count_min, ga.node_count_max),
        "max_child_count": ga.max_child_count,
        "attr_count_range": (ga.attr_count_min, ga.attr_count_max),
        "text_len_range": (ga.text_len_min, ga.text_len_max),
        "tail_len_range": (ga.tail_len_min, ga.tail_len_max),
    }
    return randomXml(treeArgs, vocabs)

def transformSample(outNode):
    """
    Transform input XML as per the changes of toy3 schema.
    
    a) All children reversed.
    b) All attributes rotated.
    c) All tail and text swapped.
    """
    # Transform all children nodes.
    for childNode in outNode:
        transformSample(childNode)

        # Swap tail and text.
        tail = childNode.tail
        childNode.tail = childNode.text
        childNode.text = tail

    # Rotate attributes.
    for i in range(len(outNode)):
        nextI = (i+1) % len(outNode)
        curChild = outNode[i]
        nextChild = outNode[nextI]

        # Swap attribs.
        temp = curChild.attrib
        curChild.attrib = nextChild.attrib
        nextChild.attrib = temp

    # Move the child out into chilren list for reversal.
    children = []
    for curChild in outNode:
        children.append(curChild)
        outNode.remove(curChild)

    # Reverse children list and insert back.
    for child in  reversed(children):
        outNode.append(child)

    return outNode
