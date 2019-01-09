import os, argparse, logging, glob
from orderedattrdict import AttrDict
from enum import Enum

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import xml.etree.ElementTree as ET

import hier2hier
from hier2hier.trainer import SupervisedTrainer
from hier2hier.models import Hier2hier, curSchemaVersion
from hier2hier.loss import Perplexity
from hier2hier.optim import Optimizer
from hier2hier.dataset import SourceField, TargetField, Hier2HierDataset
from hier2hier.evaluator import Predictor
from hier2hier.util.checkpoint import Checkpoint
from hier2hier.util import str2bool, str2bool3, levelDown

class AppMode(Enum):
    Generate=0
    Train=1
    Evaluate=2

def loadConfig(mode):
    # For usage help, issue with argument --help.
    parser = argparse.ArgumentParser()

    # Basic load/store settings.
    parser.add_argument('--domain', action='store', dest='domain', default="toy2",
                        help='The app domain to use.')
    parser.add_argument('--inputs_root_dir', action='store', dest='inputs_root_dir', default="data/inputs/",
                        help='Path to folder containing dev, train and test input data folders.')
    parser.add_argument('--training_dir', action='store', dest='training_dir', default='./data/training/',
                        help='Path to experiment directory.')
    parser.add_argument("--run", action='store', dest='run', default = None, type = int,
                        help="Index of the run that should be operated upon.")
    parser.add_argument('--resume', default=None, type=str2bool3,
                        help='Indicates if training has to be resumed from the latest checkpoint')

    # Testing data folders.
    if mode == AppMode.Evaluate:
        parser.add_argument('--test_dataset', default="test", type=str,
                        help='Dataset(test/dev/train) to use for evaluation.')
    # Some config defaults depend on the appConfig. So, peeking into appConfig, before configuring the rest.
    basicAppConfig, _ = parser.parse_known_args()
    postProcessAppConfig(basicAppConfig, mode)

    # Domain customizable load/store settings.
    parser.add_argument("--checkpoint_every", type = int, default = 100,
                        help="Number of epochs after which we take a checkpoint.")
    parser.add_argument("--input_select_percent", type = float, default = None,
                        help="Percentage of inputs actually to be selected for training. This helps in training"
                             + " with smaller dataset that what all is available.")

    # Randomizaion settings.
    parser.add_argument('--random_seed', dest='random_seed',
                        default=None,
                        help='Random seed to use before start of the training.')

    # Various logging and debug settings.
    parser.add_argument('--log_level', dest='log_level',
                        default='info',
                        help='Logging level.')
    parser.add_argument("--tensorboard",
                        type=int, default=0,
                        help="Frequency of logging data into tensorboard. Set to 0 to disable.")
    parser.add_argument("--profile",
                        type=str2bool, default=False,
                        help="Set to true to enable profiling info printing mode.")
    parser.add_argument("--runtests",
                        type=str2bool, default=False,
                        help="Set to true to enable unit testing of components.")

    # Build args needed during training.
    parser.add_argument("--epochs", type = int, default = 400,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type = int, default = 1000,
                        help="Batch size for training.")
    parser.add_argument("--num_samples", type = int, default = None,
                        help="Number of samples to train on.")
    parser.add_argument("--print_every", type = int, default = 10,
                        help="Print progress information, after every so many batches.")

    # XML Schema params.
    parser.add_argument("--max_node_count", type = int, default = None,
                        help="Maximum number of nodes in an XML file.")
    parser.add_argument("--total_attrs_count", type = int, default = None,
                        help="Total number of known attributes in the schema.")
    parser.add_argument("--value_symbols_count", type = int, default = None,
                        help="Total number of symbols used in attribute value strings.")
    parser.add_argument("--max_node_fanout", type = int, default = None,
                        help="Maximum connectivity fanout of an XML node.")
    parser.add_argument("--max_node_text_len", type = int, default = None,
                        help="Maximum length of text attribute in a node.")
    parser.add_argument("--max_attrib_value_len", type = int, default = None,
                        help="Maximum length of any text attribute value in a node.")
    parser.add_argument("--max_output_len", type = int, default = None,
                        help="Maximum length of the output file.")

    # Size meta-parameters of the generated neural network.
    parser.add_argument("--node_text_vec_len", type = int, default = 192,
                        help="Length of encoded vector for node text.")
    parser.add_argument("--attrib_value_vec_len", type = int, default = 64,
                        help="Length of an encoded attribute value.")
    parser.add_argument("--node_info_propagator_stack_depth", type = int, default = 12,
                        help="Depth of the graph layer stack. This determines the number of "
                        + "hops that information would propagate in the graph inside nodeInfoPropagator.")
    parser.add_argument("--propagated_info_len", type = int, default = 256,
                        help="Length of node information vector, when being propagated.")
    parser.add_argument("--output_decoder_stack_depth", type = int, default = 1,
                        help="Stack depth of node decoder.")
    parser.add_argument("--output_decoder_state_width", type = int, default = 200,
                        help="Width of GRU cell in output decoder.")

    # Other meta-parameters for training the neural network.
    parser.add_argument("--input_dropout_p", type = float, default = None if basicAppConfig.resume else 0.1,
                        help="Input dropout probability.")
    parser.add_argument("--dropout_p", type = float, default = None if basicAppConfig.resume else 0.1,
                        help="Dropout probability.")
    parser.add_argument("--use_attention", type = int, default = True,
                        help="Use attention while selcting most appropriate.")
    parser.add_argument("--teacher_forcing_ratio", type = int, default = 0.50,
                        help="Teacher forcing ratio to using during decoder training.")
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate to use during training.")
    parser.add_argument('--clip_gradient', type=float, default=None,
                        help='gradient clipping')
    parser.add_argument("--disable_batch_norm", type = str2bool, default = False,
                        help="Disable batch norm. Needed for running some tests.")
    if mode == AppMode.Evaluate:
        parser.add_argument("--beam_count", type = int, default = None,
                        help="Number of beams to use when decoding. Leave as None for not using beam decoding.")

    # Parse args to build app config dictionary.
    appConfig = parser.parse_args()

    # Post process app config.
    postProcessAppConfig(appConfig, mode)

    # Spin out model arguments from app configuration.
    modelArgs = levelDown(appConfig,
            "modelArgs",
            [
                # XML Schema limits.
                "max_node_count", "total_attrs_count", "value_symbols_count", "max_node_fanout",
                "max_node_text_len", "max_attrib_value_len", "max_output_len",

                # Size meta-parameters of the generated neural network.
                "node_text_vec_len", "attrib_value_vec_len", "node_info_propagator_stack_depth",
                "propagated_info_len", "output_decoder_stack_depth", "output_decoder_state_width",

                # Other meta-parameters of the generated neural network.
                "input_dropout_p", "dropout_p", "use_attention", "teacher_forcing_ratio", "learning_rate",
                "clip_gradient", "disable_batch_norm",
            ]
        )
    return appConfig, modelArgs

def postProcessAppConfig(appConfig, mode):
    """
        Post proces appConfig.
        Build derived configuration items and fix some of the appConfig defaults, where necessary.
    """
    # Restructure appConfig to constitute a configuration hierarchy.
    appConfig.debug = levelDown(appConfig, "debug", ["tensorboard", "profile", "runtests"])

    # Training, validation and evaluation data folders.
    appConfig.input_path = appConfig.inputs_root_dir + appConfig.domain + "/"
    if mode == AppMode.Train:
        appConfig.train_path = appConfig.input_path + "train/"
        appConfig.dev_path = appConfig.input_path + "dev/"
    elif mode == AppMode.Evaluate:
        appConfig.test_path = appConfig.input_path + appConfig.test_dataset + "/"

    # Training run folders.
    if not os.path.isabs(appConfig.training_dir):
        appConfig.training_dir = os.path.join(os.getcwd(), appConfig.training_dir)
    allRunsFolder = 'runFolders/'

    # By default, we are willing to create in training mode. OW, not.
    if mode == AppMode.Train:
        appConfig.create = True
    else:
        appConfig.create = False

    if appConfig.resume is True:
        # Explicit resume requested. Cannot create.
        appConfig.create = False
    elif appConfig.resume is None:
        appConfig.resume = True

    # Identify runFolder and runIndex
    (
        appConfig.runFolder,
        appConfig.run,
        appConfig.resume,
    ) = getRunFolder(
                appConfig.training_dir,
                allRunsFolder,
                runIndex=appConfig.run,
                resume=appConfig.resume,
                create=appConfig.create,
                suffix="{0}_{1}".format(appConfig.domain, curSchemaVersion))

    # Identify last checkpoint
    (
        appConfig.checkpointEpoch,
        appConfig.checkpointStep,
        appConfig.checkpointFolder
    ) = getLatestCheckpoint(appConfig.training_dir, appConfig.runFolder)

    # If runFolder exists but checkpoint folder doesn't, we still can't resume.
    if appConfig.checkpointFolder is None:
        appConfig.resume = False

def getRunFolder(dataFolderPath,
        allRunsFolder,
        runIndex=None,
        resume=True,
        create=True,
        suffix=""):
    """
    Finds all run folders inside the parent folder passed.
    if runIndex is passed, then returns the folder with specified run index.
    if runIndex is not passed, then returns the folder with latest run index.
    """
    existingRunFolders = list(glob.glob(dataFolderPath + allRunsFolder + "run." + "[0-9]"*5 + ".*"))
    existingRunFolders = [fullFolderPath for fullFolderPath in existingRunFolders if os.listdir(fullFolderPath)]
    existingRunFolders = [fullFolderPath[len(dataFolderPath):] for fullFolderPath in existingRunFolders]
    runIndexToFolderMap = { int(os.path.basename(runFolder)[4:9]):runFolder.replace("\\", "/")+"/"
                           for runFolder in existingRunFolders }

    compatibleRunIndices = [runIndex for runIndex, runFolder in runIndexToFolderMap.items()
                            if runFolder.endswith(".{0}/".format(suffix))]

    if runIndex is None:
        # Pick a runIndex that we can use.
        if resume:
            # Pick the latest compatible resume-able run.
            if compatibleRunIndices:
               runIndex = max(compatibleRunIndices)
        elif runIndexToFolderMap:
            # Pick the next run index.
            runIndex = max(runIndexToFolderMap.keys())+1
        else:
            # First run index
            runIndex = 0

    if runIndex is not None:
        # We have a run index to try.
        if runIndex not in runIndexToFolderMap:
            # Run index does not currently exist.
            if create:
                runFolder = "{0}run.{1:0>5}.{2}/".format(allRunsFolder, runIndex, suffix)
                os.makedirs(dataFolderPath + runFolder, exist_ok=True)
                return runFolder, runIndex, False
            else:
                raise FileNotFoundError("Run folder with specified index doesn't exist.")
        else:
            if runIndex in compatibleRunIndices:
                return runIndexToFolderMap[runIndex], runIndex, True
            else:
                raise ValueError("Specified run index incompatible with specified suffix.")
    else:
        if create:
            runIndex = max(runIndexToFolderMap.keys()) + 1 if runIndexToFolderMap else 0
            runFolder = "{0}run.{1:0>5}.{2}/".format(allRunsFolder, runIndex, suffix)
            os.makedirs(dataFolderPath + runFolder, exist_ok=True)
            return runFolder, runIndex, False
        else:
            raise FileNotFoundError("No run exists. To create a new run, set create=True.")

def getLatestCheckpoint(dataFolderPath, runFolder):
    """
    Identifies the latest model weights checkpoint that we can use to resume an interrupted
    indexing operation.
    """
    existingCheckpoints = list(glob.glob(dataFolderPath + runFolder + "Chk*/output_vocab.pt"))
    if existingCheckpoints:
        latestCheckpoint = max(existingCheckpoints)
        checkpointFolder = os.path.basename(os.path.dirname(latestCheckpoint))
        stepDotEpochStr = checkpointFolder[len("Chk"):]
        checkpointEpoch, checkpointStep = [int(item) for item in stepDotEpochStr.split(".")]
        return checkpointEpoch, checkpointStep, checkpointFolder
    else:
        return -1, -1, None


