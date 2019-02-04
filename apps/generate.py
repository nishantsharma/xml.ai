"""
Entry point for the application which generates simulated XML test datasets for any supported "domains".
By default, results are saved under data/inputs/<domainId>. Check supported domains under the folder domains/.

Don't call directly. Use ./scripts/generate.sh <domainId>

To see command line options, run ./scripts/generate.sh <domainId> --help
"""

from __future__ import print_function
import argparse
import os, copy, shutil, random, string, json, logging

import xml.etree.ElementTree as ET
from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

# Obtain app configuration object.
appConfig, generatorArgs = loadConfig(AppMode.Generate)

from importlib import import_module
domainModule = import_module("domains." + appConfig.domain + ".generate")

# Setup loggingLOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
inputsFolder = appConfig.inputs_root_dir + appConfig.domain + "/"
os.makedirs(inputsFolder, exist_ok=True)
logging.basicConfig(
        filename=inputsFolder + "generation.log",
        format=LOG_FORMAT,
        level=getattr(logging, appConfig.log_level.upper()))

# Log config info.
logging.info("Application Config: {0}".format(json.dumps(vars(appConfig), indent=2)))
logging.info("Unprocessed Model Arguments: {0}".format(json.dumps(generatorArgs, indent=2)))

# Generate commonData.
commonData = domainModule.generateCommon(appConfig, generatorArgs)

def generate_dataset(rootFolder, datasetName, generatorArgs, commonData, treeCount):
    """
    Generates input and output XML files for toy1 dataset.
    """
    path = os.path.join(rootFolder, datasetName)
    if not os.path.exists(path):
        os.mkdir(path)

    # generate data files in loop.
    for index in range(treeCount):
        inXml = domainModule.generateSample(generatorArgs, commonData)
        outXml = copy.deepcopy(inXml)
        domainModule.transformSample(outXml.getroot())

        # Create the generated XML.
        dataInPath = os.path.join(path, 'dataIn_{0}.xml'.format(index))
        inXml.write(dataInPath)

        # Save the transformed XML.
        dataOutPath = os.path.join(path, 'dataOut_{0}.xml'.format(index))
        outXml.write(dataOutPath)

# Generate in/out samples.
generate_dataset(inputsFolder, 'train', generatorArgs, commonData, int(0.80*generatorArgs.count))
generate_dataset(inputsFolder, 'dev', generatorArgs, commonData, int(0.10*generatorArgs.count))
generate_dataset(inputsFolder, 'test', generatorArgs, commonData, int(0.10*generatorArgs.count))


