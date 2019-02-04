from __future__ import print_function
import os, sys, shutil, random
import argparse, io 
from hier2hier.dataset.xsd2xml import GenXML
import xml.etree.ElementTree as ET

generatorArgsDefaults = {
   "xsdfile": "./domains/toy2/schema.xsd",
   "element": "PurchaseOrder",
   "enable_choice": False,
}

def addArguments(parser, defaultArgs):
    parser.add_argument("-s", "--schema", dest="xsdfile", default=defaultArgs.xsdfile,
                        help="select the xsd used to generate xml")
    parser.add_argument("-e", "--element", dest="element", default=defaultArgs.element,
                        help="select an element to dump xml")
    parser.add_argument("-c", "--choice",
                        action="store_true", dest="enable_choice", default=defaultArgs.enable_choice,
                        help="enable the <choice> mode")
    return parser

def postProcessArguments(args):
    return args 

def generateCommon(appConfig, generatorArgs):
    generator = GenXML(generatorArgs.xsdfile, generatorArgs.element, generatorArgs.enable_choice)
    return generator 

def generateSample(generatorArgs, generator):
    # Generate the XML.
    xmlStream = io.StringIO()
    generator.run(xmlStream)

    # Parse and return the XML.
    xmlStream.seek(0)
    xmlTree = ET.parse(xmlStream)
    return xmlTree

def transformSample(xmlTree):
    """
    Transforms XML tree by swapping ShipTo and BillTo addresses.
    """
    root = xmlTree.getroot()
    shipToName = root.find("ShipTo").find("name")
    billToName = root.find("BillTo").find("name")
    temp = shipToName.text
    shipToName.text = billToName.text
    billToName.text = temp
    return xmlTree
