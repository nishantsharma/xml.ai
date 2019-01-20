import os, sys, argparse, logging, json, random
from orderedattrdict import AttrDict
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import torch.nn.utils.rnn as rnn

import xml.etree.ElementTree as ET

import hier2hier
from hier2hier.trainer import SupervisedTrainer
from hier2hier.models import Hier2hier
from hier2hier.models.hier2hierBatch import batchDataUnitTest
from hier2hier.loss import Perplexity
from hier2hier.optim import Optimizer
from hier2hier.dataset import SourceField, TargetField, Hier2HierDataset, Hier2HierIterator, GeneratedXmlDataset
from hier2hier.evaluator import Predictor
from hier2hier.util.checkpoint import Checkpoint
from hier2hier.util import str2bool
from hier2hier.models.attentionSpotlight import attentionSpotlightUnitTest

from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

def hier2hierBatchTest(testNo, appConfig, modelArgs, device):
    # Instantiate trainer object.
    trainer = SupervisedTrainer(deepcopy(appConfig), deepcopy(modelArgs), device)

    # Create test data.
    generatorArgs = {
        "node_count_range": (3, 10),
        "max_child_count": 4,
        "taglen_range": (0, 5),

        "attr_count_range": (0, 3),
        "attr_len_range": (0, 5),
        "attr_value_len_range": (0, 20),

        "text_len_range": (-4, 7),
        "tail_len_range": (-20, 10),
    }
    sampleCount = 5
    test_data = GeneratedXmlDataset((sampleCount, generatorArgs), fields=trainer.fields)

    # Run tests
    batchDataUnitTest(trainer, test_data)

def smallDataTest(testNo, appConfig, modelArgs, device):
    # Instantiate trainer object.
    trainer = SupervisedTrainer(deepcopy(appConfig), deepcopy(modelArgs), device)

    # Create test data.
    generatorArgs = {
        "node_count_range": (1, 2),
        "max_child_count": 4,
        "taglen_range": (0, 5),

        "attr_count_range": (0, 3),
        "attr_len_range": (0, 5),
        "attr_value_len_range": (0, 20),

        "text_len_range": (1, 7),
        "tail_len_range": (-20, 10),
    }
    sampleCount = random.randint(1, 3)
    test_data = GeneratedXmlDataset((sampleCount, generatorArgs), fields=trainer.fields)

    # Run tests.
    trainer.train(test_data)

def noInteractionTest(testNo, appConfig, modelArgs, device):
    # Instantiate trainer object.
    modelArgs = deepcopy(modelArgs)
    appConfig = deepcopy(appConfig)
    modelArgs.spotlightThreshold = -1000000 # Pick everything.
    modelArgs.disable_batch_norm = True
    modelArgs.dropout_p = 0
    modelArgs.input_dropout_p = 0
    modelArgs.teacher_forcing_ratio = 0
    trainer = SupervisedTrainer(appConfig, modelArgs, device)

    # Create test data.
    generatorArgs = {
        "node_count_range": (5, 10),
        "max_child_count": 4,
        "taglen_range": (2, 5),

        "attr_count_range": (0, 4),
        "attr_len_range": (2, 5),
        "attr_value_len_range": (1, 20),

        "text_len_range": (-10, 7),
        "tail_len_range": (-20, 10),
    }
    sampleCount = 50
    test_data = GeneratedXmlDataset((sampleCount, generatorArgs), fields=trainer.fields)
    trainer.load(test_data)
    examples = test_data.examples

    exampleToWatch = examples[0]
    remainingExamples = examples[1:]
    resultsToWatch = []
    prevResult = None
    prevGraphNodeCount = None
    for trial in range(20):
        chosenExamples = random.sample(remainingExamples, 5)
        watchPosition = random.randint(0, len(chosenExamples))
        print("Selected watch position ", watchPosition)
        chosenExamples.insert(watchPosition, exampleToWatch)
        assert(chosenExamples[watchPosition] == exampleToWatch)
        test_data_section = GeneratedXmlDataset(chosenExamples, fields=trainer.fields)

        batch_iterator = Hier2HierIterator(
            preprocess_batch=trainer.model.preprocess_batch,
            dataset=test_data_section, batch_size=len(test_data_section),
            sort=False, shuffle=False, sort_within_batch=False,
            device=device,
            repeat=False,
            )
        batch_generator = batch_iterator.__iter__(unitTesting=True)
        test_data_batch = list(batch_generator)[0]

        dataDebugStages = trainer.model(test_data_batch, debugDataStages=True)

        curGraphNodeCount = len([gni for gni, toi in enumerate(test_data_batch.gni2Toi) if toi==watchPosition])
        if prevGraphNodeCount is not None:
            assert(curGraphNodeCount == prevGraphNodeCount)
        prevGraphNodeCount = curGraphNodeCount

        curResult = [ (stageName, dataDebugStage[watchPosition]) for (stageName, dataDebugStage) in dataDebugStages]
        outputLenToWatch=test_data_batch.targetOutputLengthsByToi[watchPosition]

        # Remove computation stages which are not relevant for exampleToWatch at watchPosition.
        _curResult = []
        for result in curResult:
            atPos = result[0].find("@")
            if atPos >= 0:
                try:
                    charIndex = int(result[0][0:atPos])
                    if charIndex >= outputLenToWatch:
                        continue
                except ValueError:
                    pass
            _curResult.append(result)
        curResult = _curResult

        if prevResult is not None:
            assert(len(curResult) == len(prevResult))
            stageCount = len(curResult)
            for stage in range(stageCount):
                assert(curResult[stage][1].shape == prevResult[stage][1].shape)
                diff = curResult[stage][1] - prevResult[stage][1]
                diffNorm = torch.norm(diff)
                print("Trial {0}. Stage {1}:{2}. Diff {3}".format(
                    trial,
                    stage,
                    (
                        curResult[stage][0]
                        if curResult[stage][0] == prevResult[stage][0]
                        else curResult[stage][0] + "/" + prevResult[stage][0]
                    ),
                    diffNorm,))
                assert(diffNorm < 1e-5)
        prevResult = curResult

def attentionSpotlightTest(testNo, appConfig, modelArgs, device):
    attentionSpotlightUnitTest()
    
def trainingProgressTest(testNo, appConfig, modelArgs, device):
    # Instantiate trainer object.
    modelArgs = deepcopy(modelArgs)
    appConfig = deepcopy(appConfig)
    modelArgs.spotlightThreshold = -1000000 # Pick everything.
    modelArgs.disable_batch_norm = True
    modelArgs.dropout_p = 0
    modelArgs.input_dropout_p = 0
    modelArgs.teacher_forcing_ratio = 0.5
    appConfig.epochs = 10
    trainer = SupervisedTrainer(appConfig, modelArgs, device)

    # Create test data.
    generatorArgs = {
        "node_count_range": (2, 6),
        "max_child_count": 4,
        "taglen_range": (2, 5),

        "attr_count_range": (0, 4),
        "attr_len_range": (2, 5),
        "attr_value_len_range": (1, 7),

        "text_len_range": (-10, 7),
        "tail_len_range": (-20, 10),
    }
    sampleCount = 5
    test_data = GeneratedXmlDataset((sampleCount, generatorArgs), fields=trainer.fields)

    # Run tests.
    trainer.train(test_data)

def hierarchyPropagatorTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def encoderPermutationTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def batchGraphConsistencyTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def spotlightTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def beamTest(testNo, appConfig, modelArgs, device):
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

    # Test the model.
    for testFunc in [
        # Training should progress for a single dataset.
        trainingProgressTest,

        # Batch data should be generated correctly.
        hier2hierBatchTest,

        # Must not crash on small data. Results must match with direct component calls.
        smallDataTest,

        # Attention Spotlight unit test.
        attentionSpotlightTest,

        # Changing other trees or permuting input trees should not change the output of
        # an individual tree.
        noInteractionTest,

        # Propagate hierarchy data.
        hierarchyPropagatorTest,

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
        for testNo in range(appConfig.repetitionCount):
            testFunc(testNo, appConfig, modelArgs, device)

main()
