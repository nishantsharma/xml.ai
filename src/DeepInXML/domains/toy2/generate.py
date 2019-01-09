from __future__ import print_function
import argparse
import os, sys
import shutil
import random
from hier2hier.dataset.xsd2xml import GenXML
import xml.etree.ElementTree as ET

def transform(xmlTree):
    root = xmlTree.getroot()
    shipToName = root.find("ShipTo").find("name")
    billToName = root.find("BillTo").find("name")
    temp = shipToName.text
    shipToName.text = billToName.text
    billToName.text = temp

def generate_dataset(generator, root, name, size):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)

    # generate data file
    for index in range(size):
        # Write the generated XML.
        dataInPath = os.path.join(path, 'dataIn_{0}.xml'.format(index))
        with open(dataInPath, 'w') as fout:
            generator.run(fout)

        # Transform the XML.
        xmlTree = ET.parse(dataInPath)
        transform(xmlTree)

        # Save the transformed XML.
        dataOutPath = os.path.join(path, 'dataOut_{0}.xml'.format(index))
        xmlTree.write(dataOutPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--schema", dest="xsdfile", required=True,
                        help="select the xsd used to generate xml")
    parser.add_argument("-e", "--element", dest="element", required=True,
                        help="select an element to dump xml")
    parser.add_argument("-c", "--choice",
                        action="store_true", dest="enable_choice", default=False,
                        help="enable the <choice> mode")
    parser.add_argument('-d', '--dir',
                        help="data directory", default="../../../data")
    parser.add_argument('-m', '--max-len',
                        help="max sequence length", default=10)
    args = parser.parse_args()

    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    toy_dir = os.path.join(data_dir, 'toy_PurchaseOrderAddrSwap')
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    generator = GenXML(args.xsdfile, args.element, args.enable_choice)

    generate_dataset(generator, toy_dir, 'train', 10000)
    generate_dataset(generator, toy_dir, 'dev', 1000)
    generate_dataset(generator, toy_dir, 'test', 1000)
