"""
Module to generate toy1 dataset.
"""

from __future__ import print_function
import argparse
import os
import shutil
import random
import string

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="data/")
parser.add_argument('--max-len', help="max sequence length", default=10)
parser.add_argument('--count', help="Total number of data entries to generate", default=1000)
args = parser.parse_args()

def generate_dataset(rootFolder, datasetName, treeCount):
    """
    Generates input and output XML files for toy1 dataset.
    """
    path = os.path.join(rootFolder, datasetName)
    if not os.path.exists(path):
        os.mkdir(path)

    srcAlphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    # generate data file
    for index in range(treeCount):
        length = random.randint(1, args.max_len)
        dataSeq = []
        dataStr = ""
        for _ in range(length):
            ch = srcAlphabet[random.randint(0, len(srcAlphabet)-1)]
            dataSeq.append(str(ord(ch)))
            dataStr += ch

        # Update data.txt.
        data_path = os.path.join(path, 'data.txt')
        with open(data_path, 'a') as fout:
            fout.write("\t".join([" ".join(dataSeq), " ".join(reversed(dataSeq))]))
            fout.write('\n')

        # Create the generated XML.
        dataInPath = os.path.join(path, 'dataIn_{0}.xml'.format(index))
        with open(dataInPath, 'w') as fout:
            fout.write("<toyrev>" + dataStr + "</toyrev>")

        # Save the transformed XML.
        dataOutPath = os.path.join(path, 'dataOut_{0}.xml'.format(index))
        with open(dataOutPath, 'w') as fout:
            fout.write("<toyrev>" + "".join(reversed(dataStr)) + "</toyrev>")

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
