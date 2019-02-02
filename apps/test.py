"""
Entry point for the application which runs a battery of tests on the model implementation.
Results are rendered on the command line.

Don't call directly. Use ./scripts/test.sh

To see command line options, run ./scripts/test.sh --help
"""

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
from hier2hier.util import str2bool, longTensor
from hier2hier.models.attentionSpotlight import attentionSpotlightUnitTest

from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

def hier2hierBatchTest(testNo, appConfig, modelArgs, device):
    # Instantiate trainer object.
    trainer = SupervisedTrainer(deepcopy(appConfig), deepcopy(modelArgs), device)

    # Create test data.
    generatorArgs = {
        "node_count_range": (3, 10),
        "max_child_count": 4,

        "tag_gen_params": (50, (0, 5)),
        "attr_gen_params": (50, (0, 5)),
        "attr_value_gen_params": (50, (0, 20)),

        "attr_count_range": (0, 3),
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

        "tag_gen_params": (50, (0, 5)),
        "attr_gen_params": (50, (0, 5)),
        "attr_value_gen_params": (50, (0, 20)),

        "attr_count_range": (0, 3),
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

        "tag_gen_params": (50, (2, 5)),
        "attr_gen_params": (50, (0, 4)),
        "attr_value_gen_params": (50, (1, 20)),

        "attr_len_range": (2, 5),
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
        batch_generator = batch_iterator.__iter__(mode=appConfig.mode)
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

def knownSpotlightTest(testNo, appConfig, modelArgs, device):
    """
    In this test, we train over toy1(node.text reversal). We use one hot encoded attention which matches
    directly with source symbol. This should test if rest of the system is working when we use hard coded attention.
    """
    # Instantiate trainer object.
    modelArgs = deepcopy(modelArgs)
    appConfig = deepcopy(appConfig)
    appConfig.train_path = appConfig.inputs_root_dir + "toy1/dev/"
    appConfig.dev_path = appConfig.inputs_root_dir + "toy1/dev/"
    modelArgs.disable_batch_norm = True
    modelArgs.dropout_p = 0
    modelArgs.input_dropout_p = 0

    appConfig.epochs = 1000
    modelArgs.learning_rate = 0.005

    def spotlightByFormula(hier2hierBatch, sampleIndexLimit, outputIndex):
        """
        For the node.text reversal dataset, this function computes the spotlight to use 
        by finding the right input symbol to focus on.
        """
        retval = []
        # Text positions to NDTLP.
        isTail=False
        for tdol in range(sampleIndexLimit):
            targetOutput = hier2hierBatch.targetOutputsByTdol[tdol]
            targetOutputLength = int(hier2hierBatch.targetOutputLengthsByTdol[tdol])

            # Get ndfo position of the input node.
            toi = int(hier2hierBatch.tdol2Toi[tdol])
            rootNode = hier2hierBatch.inputs[toi].getroot()
            ndfo = hier2hierBatch.node2Ndfo[rootNode]

            # Compute gndtol of the input XML node.
            # Computed as follows: inputs[toi] -> getroot() -> node2Ndfo -> ndfo2Gni -> gni2Gndtol.
            nodeGni = hier2hierBatch.ndfo2Gni[ndfo]
            nodeGndtol = hier2hierBatch.gni2Gndtol[nodeGni]

            if outputIndex < 1 + len("<toyrev>"):
                # We are in the opening tag portion of the output.
                # Return gndtol of the XML node.
                spotIndex = nodeGndtol
            else:
                # Compute input and output char index within node.text, where the current output pointer is. 
                outputStrIndex = outputIndex - len("<toyrev>") - 1 # Don't forget <sos>.
                outputStrLength = targetOutputLength - len("<toyrev>") - len("</toyrev>") - 2 # Don't forget <sos> and <eos>.
                inputStrIndex = outputStrLength - outputStrIndex - 1

                if inputStrIndex < 0:
                    # We are currently in the closing tag portion of the output.
                    # Return gndtol of the XML node.
                    spotIndex = nodeGndtol
                else:
                    # Get the raw char stored at input.
                    ch = hier2hierBatch.inputs[toi].getroot().text[inputStrIndex]

                    # Make sure that input is encoded correctly.
                    ndtx2 = hier2hierBatch.ndfo2Ndtl2[isTail][ndfo]
                    ndtxp2 = hier2hierBatch.ndtxTuple2Ndtlp2[isTail][(ndtx2, inputStrIndex)]
                    encodedInputSymbol = int(hier2hierBatch.encodedTextByTtDLP.data[ndtxp2])
                    inputVocab = hier2hierBatch.torchBatch.dataset.fields["src"].vocabs.text
                    assert(ch == inputVocab.itos[encodedInputSymbol])

                    # Compute spot index from gndtol of output character position.
                    charGni = hier2hierBatch.ndttp2Gni[ndtxp2]
                    charGndtol = hier2hierBatch.gni2Gndtol[charGni]
                    spotIndex = charGndtol

                    # Make sure that target output is encoded correctly.
                    targetOutput = hier2hierBatch.targetOutputsByTdol[tdol]
                    outputVocab = hier2hierBatch.torchBatch.dataset.fields["tgt"].vocab
                    assert(ch == outputVocab.itos[targetOutput[outputIndex]])

            retval.append(spotIndex)
        return torch.LongTensor(retval)

    # Build trainer with a model which will use formula for spotlight.
    trainer = SupervisedTrainer(appConfig, modelArgs, device, spotlightByFormula)

    # Load training and dev dataset.
    training_data = Hier2HierDataset(baseFolder=appConfig.train_path, fields=trainer.fields, selectPercent=appConfig.input_select_percent)
    dev_data = Hier2HierDataset(baseFolder=appConfig.dev_path, fields=trainer.fields, selectPercent=appConfig.input_select_percent)

    # Train the model.
    trainer.train(training_data, dev_data=dev_data)

    # Train once more.
    import pdb;pdb.set_trace()
    appConfig.epochs = 2000
    trainer.model.outputDecoder.spotlightByFormula=None
    trainer.train(training_data, dev_data=dev_data)

def spotlightTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

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

        "tag_gen_params": (50, (2, 5)),
        "attr_gen_params": (50, (2, 5)),
        "attr_value_gen_params": (50, (1, 7)),

        "attr_count_range": (0, 4),
        "text_len_range": (-10, 7),
        "tail_len_range": (-20, 10),
    }
    sampleCount = 5
    test_data = GeneratedXmlDataset((sampleCount, generatorArgs), fields=trainer.fields)

    # Run tests.
    trainer.train(test_data)
    import pdb;pdb.set_trace()

def hierarchyPropagatorTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def encoderPermutationTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def batchGraphConsistencyTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def SpotNeighborsExplorerTest(testNo, appConfig, modelArgs, device):
    from hier2hier.models.spotNeighborsExplorer import SpotNeighborsExplorer
    nodeCount = random.randint(1, 100)
    maxNbrs = random.randint(1, nodeCount/2)
    nbrs = torch.randint(nodeCount, (nodeCount, maxNbrs), device=device)
    nbrCounts = torch.randint(maxNbrs, (nodeCount,), device=device)
    for node in range(nodeCount):
        for nbr in range(int(nbrCounts[node]), maxNbrs):
            nbrs[node, nbr] = -1
    graph = (nbrs, nbrCounts)

    explorerUnderTest = SpotNeighborsExplorer(device=device)
    explorerToMatch = SpotNeighborsExplorer(impl_selection="python", device=device)

    startAtiveSetCount = random.randint(0, int(nodeCount/3))
    activeSetIn = longTensor(
        sorted(random.sample(range(nodeCount), startAtiveSetCount)),
        device=device
    )
    alreadySeenSetIn = activeSetIn.clone()
    while activeSetIn.shape[0]:
        alreadySeenOut1, activeSetOut1 = explorerToMatch(graph, alreadySeenSetIn, activeSetIn)
        alreadySeenOut2, activeSetOut2 = explorerUnderTest(graph, alreadySeenSetIn, activeSetIn)

        assert(set(alreadySeenOut1.tolist()) == set(alreadySeenOut2.tolist()))
        assert(len(alreadySeenOut1.tolist()) == len(alreadySeenOut2.tolist()))
        assert(set(activeSetOut1.tolist()) == set(activeSetOut2.tolist()))
        assert(len(activeSetOut1.tolist()) == len(activeSetOut2.tolist()))

        alreadySeenSetIn = alreadySeenOut1
        activeSetIn = activeSetOut1

def beamTest(testNo, appConfig, modelArgs, device):
    raise NotImplementedError()

def main():
    # Obtain app configuration object.
    appConfig, modelArgs = loadConfig(AppMode.Test)

    # Setup logging
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    os.makedirs(appConfig.runFolder, exist_ok=True)
    logging.basicConfig(filename=appConfig.runFolder + "training.log", format=LOG_FORMAT, level=getattr(logging, appConfig.log_level.upper()))

    # Log config info.
    logging.info("Application Config: {0}".format(json.dumps(vars(appConfig), indent=2)))
    logging.info("Unprocessed Model Arguments: {0}".format(json.dumps(modelArgs, indent=2)))

    # Pick the device, preferably GPU where we run our application.
    device = torch.device("cuda") if torch.cuda.is_available() else None

    # Test the model.
    for testFunc in [
        # Spot neighbors explorer.
        SpotNeighborsExplorerTest,

        # When a static spotlight moves as expected.
        knownSpotlightTest,

        # Batch data should be generated correctly.
        hier2hierBatchTest,

        # Must not crash on small data. Results must match with direct component calls.
        smallDataTest,

        # Attention Spotlight unit test.
        attentionSpotlightTest,

        # Changing other trees or permuting input trees should not change the output of
        # an individual tree.
        noInteractionTest,

        # Training should progress for a single dataset.
        trainingProgressTest,

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
