"""
Module to generate toy3 dataset.
"""

from __future__ import print_function
import argparse
import os, copy, shutil, random, string

import xml.etree.ElementTree as ET
from hier2hier.dataset.dataset import generateXml

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="data/")
parser.add_argument('--max-len', help="max sequence length", default=10)
parser.add_argument('--count', help="Total number of data entries to generate", default=1000)
args = parser.parse_args()

def transformXml(outNode):
    """
    Transform input XML as per the changes of toy3 schema.
    """
    # Transform all children nodes.
    for childNode in outNode:
        transformXml(childNode)

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


def generate_dataset(rootFolder, datasetName, treeCount):
    """
    Generates input and output XML files for toy1 dataset.
    """
    path = os.path.join(rootFolder, datasetName)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create test data.
    generatorArgs = {
        "node_count_range": (2, 5),
        "max_child_count": 3,

        "tag_gen_params": (30, (1, 3)), # (tag Count, (min tag length, max tag length))
        "attr_gen_params": (20, (1, 3)), # (attr Count, (min attr length, max attr length))
        "attr_value_gen_params": (30, (1, 3)), # (attr value Count, (min attr value length, max attr value length))

        "attr_count_range": (0, 2),
        "text_len_range": (-10, 3),
        "tail_len_range": (-20, 3),
    }

    # generate data files in loop.
    for index in range(treeCount):
        inXml = generateXml(generatorArgs)
        outXml = copy.deepcopy(inXml)
        transformXml(outXml.getroot())

        # Create the generated XML.
        dataInPath = os.path.join(path, 'dataIn_{0}.xml'.format(index))
        inXml.write(dataInPath)

        # Save the transformed XML.
        dataOutPath = os.path.join(path, 'dataOut_{0}.xml'.format(index))
        outXml.write(dataOutPath)

if __name__ == '__main__':
    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    toy_dir = data_dir
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    generate_dataset(toy_dir, 'train', int(0.80*args.count))
    generate_dataset(toy_dir, 'dev', int(0.10*args.count))
    generate_dataset(toy_dir, 'test', int(0.10*args.count))
