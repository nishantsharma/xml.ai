import os, sys, argparse, logging, json
from orderedattrdict import AttrDict

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import xml.etree.ElementTree as ET

import hier2hier
from hier2hier.trainer import SupervisedTrainer
from hier2hier.models import Hier2hier
from hier2hier.loss import Perplexity
from hier2hier.optim import Optimizer
from hier2hier.dataset import SourceField, TargetField, Hier2HierDataset, GeneratedXmlDataset
from hier2hier.evaluator import Predictor
from hier2hier.util.checkpoint import Checkpoint
from hier2hier.util import str2bool

from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

def smallDataTest(appConfig, trainer):
    generatorArgs = {
        "node_count_range": (1, 2),
        "max_child_count": 4,
        "taglen_range": (0, 5),

        "attr_count_range": (0, 1),
        #"attr_len_range": (0, 5),
        #"attr_value_len_range": (0, 20),

        "text_len_range": (1, 7),
        "tail_len_range": (-20, 10),
    }
    for i in range(appConfig.repetitionCount):
        test_data = GeneratedXmlDataset(1, generatorArgs, fields=trainer.fields)
        trainer.train(test_data)


def singleElementTest(appConfig, trainer):
    raise NotImplementedError()

def fiveTreesTest(appConfig, trainer):
    raise NotImplementedError()

def permutationTest(appConfig, trainer):
    raise NotImplementedError()

def encoderPermutationTest(appConfig, trainer):
    raise NotImplementedError()

def batchGraphConsistencyTest(appConfig, trainer):
    raise NotImplementedError()

def spotlightTest(appConfig, trainer):
    raise NotImplementedError()

def beamTest(appConfig, trainer):
    raise NotImplementedError()

def main():
    # Obtain app configuration object.
    appConfig, modelArgs = loadConfig(AppMode.Test)

    # Setup logging
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(filename=appConfig.runFolder + "training.log", format=LOG_FORMAT, level=getattr(logging, appConfig.log_level.upper()))

    # Log config info.
    logging.info("Application Config: {0}".format(json.dumps(vars(appConfig), indent=2)))
    logging.info("Unprocessed Model Arguments: {0}".format(json.dumps(modelArgs, indent=2)))

    # Pick the device, preferably GPU where we run our application.
    device = torch.device("cuda") if torch.cuda.is_available() else None

    # Trainer object is requred to
    trainer = SupervisedTrainer(appConfig, modelArgs, device)

    # Test the model.
    for testFunc in [
        # Must not crash on small data. Results must match with direct component calls.
        smallDataTest,

        # Output for a single tree should match results created without using
        # batch pre-processing.
        singleElementTest,

        # Tree position changes should not affect the output of an individual tree.
        # That should match with singleton call. 
        fiveTreesTest,

        # Permuting the trees and node-attributes should not change anything.
        permutationTest,

        # Permuting the nodes should not change encoding output.
        encoderPermutationTest,

        # Construct XML from the graph.
        batchGraphConsistencyTest,

        # Next iteration of spotlight should have all good reachable nodes and no bad
        # reachable node.
        spotlightTest,

        # Single beam results should match with multi-beam.
        # Multi-beam results should always be better than single beam.
        beamTest,
    ]:
        testFunc(appConfig, trainer)

main()
