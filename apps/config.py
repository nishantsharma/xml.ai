"""
Configuration subsystem for the applications(train/test/evaluate).
Two types of configuration objects are created here.

appConfig: Configuration related with the specific launch of the application.

modelArgs: These configuration items determine the nature of the deep learning 
    model generated. They are unlikely to change across application runs.
"""
import os, argparse, logging, glob, random, re
from orderedattrdict import AttrDict
from importlib import import_module

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import xml.etree.ElementTree as ET

import hier2hier
from hier2hier.trainer import SupervisedTrainer, defaultSchemaVersion
from hier2hier.models import Hier2hier
from hier2hier.util import str2bool, str2bool3, levelDown, AppMode


# Global AppConfig defaults:
#       Unless an app config item is overridden at the command line level or
#       at the domain level, it is picked up from here.
appConfigGlobalDefaults = {
    "checkpoint_every": 100,
}

# Global modelArgs defaults:
#       Unless a modelArgs config item is overridden at the command line level or
#       at the domain level, it is picked up from here.
modelArgsGlobalDefaults = {
    # XML Schema limits.
    "schemaVersion": None,
    "max_node_count": None,
    "node_type_count": None,
    "total_attrs_count": None,
    "value_symbols_count": None,
    "max_node_fanout": None,
    "max_node_text_len": None,
    "max_attrib_value_len": None,
    "max_output_len": None,

    # Size meta-parameters of the generated neural network.
    "node_text_vec_len": 192,
    "attrib_value_vec_len": 64,
    "node_info_propagator_stack_depth": 3,
    "propagated_info_len": 256,
    "attentionSubspaceVecLen": 20,
    "output_decoder_stack_depth": 1,
    "output_decoder_state_width": 200,

    # Other meta-parameters of the generated neural network.
    "input_dropout_p": 0.1,
    "dropout_p": 0.1,
    "use_attention": True,
    "teacher_forcing_ratio": 0.50,
    "learning_rate": 0.001,
    "clip_gradient": None,
    "disable_batch_norm": None,
    "enableSpotlight": True,
    "spotlightThreshold": 0.001,
    "useSrcPtr": True,
}

# Global generatorArgs defaults:
#       Unless a generatorArgs config item is overridden at the command line level or
#       at the domain level, it is picked up from here.
generatorArgsGlobalDefaults = {
    "count": 10000,
}

def loadConfig(mode):
    """
    Build config objects for application launch instance(appConfig) and model arguments.
    Configuration is built by merging:
        Options specified at the command line(highest priority).
        Domain specific defaults specified in the domain directory(next priority).
        Global defaults defined above.

    Returns:
        Tuple of appConfig and modelArgs.
    """
    def __basic_arguments_parser(add_help):
        """
        Creates a command line parser with a minimal set of folder level arguments.
        """
        parser = argparse.ArgumentParser(add_help=add_help)

        # Basic load/store settings.
        parser.add_argument('--domain', action='store', dest='domain', default="toy2",
                            help='The app domain to use.')
        parser.add_argument('--inputs_root_dir', action='store', dest='inputs_root_dir', default="data/inputs/",
                            help='Path to folder containing dev, train and test input data folders.')
        if mode in [AppMode.Train, AppMode.Evaluate]:
            parser.add_argument('--training_dir', action='store', dest='training_dir',
                                default='./data/training/' if mode!=AppMode.Test else "./data/testing/",
                                help='Path to experiment directory.')
            parser.add_argument("--run", action='store', dest='run', default = None, type = int,
                                help="Index of the run that should be operated upon.")

            parser.add_argument("--schemaVersion", type = int,
                                default = None,
                                help="Maximum number of nodes in an XML file.")

            parser.add_argument('--resume', default=False if mode==AppMode.Test else None, type=str2bool3,
                                help='Indicates if training has to be resumed from the latest checkpoint')

        # Testing data folders.
        if mode == AppMode.Evaluate:
            parser.add_argument('--test_dataset', default="test", type=str,
                            help='Dataset(test/dev/train) to use for evaluation.')

        return parser

    # Some config defaults depend on the appConfig. So, peeking into appConfig, before configuring the rest
    basicAppConfig, _ = __basic_arguments_parser(False).parse_known_args()
    postProcessAppConfig(basicAppConfig, mode)

    # Get domain defaults and merge them.
    domainModule = import_module("domains." + basicAppConfig.domain)
    appConfigDefaults = AttrDict({ **appConfigGlobalDefaults, **domainModule.appConfigDefaults})
    if mode == AppMode.Generate:
        domainGeneratorModule = import_module("domains." + basicAppConfig.domain + ".generate")
        generatorArgsDefaults = AttrDict({ **generatorArgsGlobalDefaults, **domainGeneratorModule.generatorArgsDefaults})
    else:
        modelArgsDefaults = AttrDict({ **modelArgsGlobalDefaults, **domainModule.modelArgsDefaults})

    # Create the parser which parses basic arguments and will also parse the entire kitchen sink, below.
    parser = __basic_arguments_parser(True)

    # Testing multiple times.
    if mode == AppMode.Test:
        parser.add_argument("--repetitionCount", type = int, default = 5,
                            help="Number of times to repeat test.")

    if mode in [AppMode.Train]:
        # Domain customizable load/store settings.
        parser.add_argument("--checkpoint_every", type = int, default = appConfigDefaults.checkpoint_every,
                            help="Number of epochs after which we take a checkpoint.")
        parser.add_argument("--print_every", type = int, default = 10,
                            help="Print progress information, after every so many batches.")

    if mode in [AppMode.Train, AppMode.Evaluate]:
        parser.add_argument("--input_select_percent", type = float, default = None,
                            help="Percentage of inputs actually to be selected for training. This helps in training"
                                 + " with smaller dataset that what all is available.")

    # Randomizaion settings.
    parser.add_argument('--random_seed', dest='random_seed',
                        type=float, default=None,
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
    if mode in [AppMode.Train, AppMode.Evaluate]:
        parser.add_argument("--runtests",
                            type=str2bool, default=False,
                            help="Set to true to enable unit testing of components.")
        parser.add_argument("--debugAttention", dest="attention",
                            type=str2bool, default=False,
                            help="Debug attention by loggnig it into tensorboard.")

    parser.add_argument("--batch_size", type = int, default = 1000,
                        help="Batch size for training.")
 
    if mode in [AppMode.Train, AppMode.Evaluate]:
        # Build args needed during training.
        parser.add_argument("--epochs", type = int,
                            default = random.randint(1, 10) if mode == AppMode.Test else 400,
                            help="Number of epochs to train for.")

        # parser.add_argument("--num_samples", type = int, default = None,
        #                    help="Number of samples to train on.")

    if mode in [AppMode.Train, AppMode.Evaluate, AppMode.Test]:
        # XML Schema params.
        parser.add_argument("--max_node_count", type = int,
                            default = modelArgsDefaults.max_node_count,
                            help="Maximum number of nodes in an XML file.")
        parser.add_argument("--total_attrs_count", type = int,
                            default = modelArgsDefaults.total_attrs_count,
                            help="Total number of known attributes in the schema.")
        parser.add_argument("--value_symbols_count", type = int,
                            default = modelArgsDefaults.value_symbols_count,
                            help="Total number of symbols used in attribute value strings.")
        parser.add_argument("--max_node_fanout", type = int,
                            default = modelArgsDefaults.max_node_fanout,
                            help="Maximum connectivity fanout of an XML node.")
        parser.add_argument("--max_node_text_len", type = int,
                            default = modelArgsDefaults.max_node_text_len,
                            help="Maximum length of text attribute in a node.")
        parser.add_argument("--max_attrib_value_len", type = int,
                            default = modelArgsDefaults.max_attrib_value_len,
                            help="Maximum length of any text attribute value in a node.")
        parser.add_argument("--max_output_len", type = int,
                            default = modelArgsDefaults.max_output_len,
                            help="Maximum length of the output file.")

        # Size meta-parameters of the generated neural network.
        parser.add_argument("--node_text_vec_len", type = int,
                            default = modelArgsDefaults.node_text_vec_len,
                            help="Length of encoded vector for node text.")
        parser.add_argument("--attrib_value_vec_len", type = int,
                            default = modelArgsDefaults.attrib_value_vec_len,
                            help="Length of an encoded attribute value.")
        parser.add_argument("--node_info_propagator_stack_depth", type = int,
                            default = modelArgsDefaults.node_info_propagator_stack_depth,
                            help="Depth of the graph layer stack. This determines the number of "
                            + "hops that information would propagate in the graph inside nodeInfoPropagator.")
        parser.add_argument("--propagated_info_len", type = int,
                            default = modelArgsDefaults.propagated_info_len,
                            help="Length of node information vector, when being propagated.")
        parser.add_argument("--output_decoder_stack_depth", type = int,
                            default = modelArgsDefaults.output_decoder_stack_depth,
                            help="Stack depth of node decoder.")
        parser.add_argument("--output_decoder_state_width", type = int,
                            default = modelArgsDefaults.output_decoder_state_width,
                            help="Width of GRU cell in output decoder.")
        parser.add_argument("--attentionSubspaceVecLen", type = int,
                            default = modelArgsDefaults.attentionSubspaceVecLen,
                            help="Vec length of subspace of attnReadyVecs and decoder hidden state"
                            + "that is used to comptue attention factors.")

        # Other meta-parameters for training the neural network.
        parser.add_argument("--input_dropout_p", type = float,
                            default = None if basicAppConfig.resume else modelArgsDefaults.input_dropout_p,
                            help="Input dropout probability.")
        parser.add_argument("--dropout_p", type = float,
                            default = None if basicAppConfig.resume else modelArgsDefaults.dropout_p,
                            help="Dropout probability.")
        parser.add_argument("--use_attention", type = int,
                            default = modelArgsDefaults.use_attention,
                            help="Use attention while selcting most appropriate.")
        parser.add_argument("--teacher_forcing_ratio", type = int,
                            default = modelArgsDefaults.teacher_forcing_ratio,
                            help="Teacher forcing ratio to using during decoder training.")
        parser.add_argument("--learning_rate", type = float,
                            default = modelArgsDefaults.learning_rate,
                            help="Learning rate to use during training.")
        parser.add_argument('--clip_gradient', type=float,
                            default=modelArgsDefaults.clip_gradient,
                            help='gradient clipping')
        parser.add_argument("--disable_batch_norm", type = str2bool,
                            default = modelArgsDefaults.disable_batch_norm,
                            help="Disable batch norm. Needed for running some tests.")
        parser.add_argument("--enableSpotlight", type = str2bool,
                            default = modelArgsDefaults.enableSpotlight,
                            help="Whether to enable spotlight, which is designed to optimize"
                                + " search.")
        parser.add_argument("--spotlightThreshold", type = float,
                            default = modelArgsDefaults.spotlightThreshold,
                            help="Threshold used to identify encoder positions to be considered"
                                + "for evaluation.")                        
        parser.add_argument("--useSrcPtr", type = str2bool,
                            default = modelArgsDefaults.useSrcPtr,
                            help="When set to true, the model is enabled to directly copy symbols from source.")

    if mode == AppMode.Evaluate:
        parser.add_argument("--beam_count", type = int,
                        default = None,
                        help="Number of beams to use when decoding. Leave as None for not using beam decoding.")

    if mode == AppMode.Generate:
        parser.add_argument('--count', help="Total number of data entries to generate",
                default=generatorArgsDefaults.count)
        parser = domainGeneratorModule.addArguments(parser, generatorArgsDefaults)

    # Parse args to build app config dictionary.
    appConfig = parser.parse_args()

    # Apply random seed.
    if appConfig.random_seed is not None:
        random.seed(appConfig.random_seed)
        torch.manual_seed(appConfig.random_seed)

    appConfig.mode = int(mode)

    if mode == AppMode.Generate:
        domainGeneratorModule.postProcessArguments(appConfig)
        generatorArgs = levelDown(appConfig, "generatorArgs", generatorArgsDefaults.keys())
        return appConfig, generatorArgs
    else:
        # Post process app config.
        postProcessAppConfig(appConfig, mode)

        # Spin out model arguments from app configuration.
        modelArgs = levelDown(appConfig, "modelArgs", modelArgsDefaults.keys())

        return appConfig, modelArgs

def postProcessAppConfig(appConfig, mode):
    """
        Post proces appConfig.
        Build derived configuration items and fix some of the appConfig defaults, where necessary.
    """
    appConfig.mode = int(mode)
    # Restructure appConfig to constitute a configuration hierarchy.
    appConfig.debug = levelDown(appConfig, "debug", ["attention", "tensorboard", "profile", "runtests"])

    # Training, validation and evaluation data folders.
    appConfig.input_path = appConfig.inputs_root_dir + appConfig.domain + "/"
    if mode == AppMode.Train:
        appConfig.train_path = appConfig.input_path + "train/"
        appConfig.dev_path = appConfig.input_path + "dev/"
    elif mode == AppMode.Evaluate:
        appConfig.test_path = appConfig.input_path + appConfig.test_dataset + "/"

    # Training run folders.
    if mode != AppMode.Generate:
        if not os.path.isabs(appConfig.training_dir):
            appConfig.training_dir = os.path.join(os.getcwd(), appConfig.training_dir)
    allRunsFolder = 'runFolders/'

    # By default, we are willing to create in training mode. OW, not.
    if mode in [AppMode.Train, AppMode.Test]:
        appConfig.create = None
    else:
        appConfig.create = False

    if mode != AppMode.Generate:
        if appConfig.resume is True:
            # Explicit resume requested. Cannot create.
            appConfig.create = False
        #elif appConfig.resume is None:
        #    appConfig.resume = True

        # Identify runFolder and runIndex
        if appConfig.schemaVersion is None:
            suffixRegex = "{0}_\d*".format(appConfig.domain)
        else:
            suffixRegex = "{0}_{1}".format(appConfig.domain, appConfig.schemaVersion)

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
                    suffixRegex=suffixRegex,
                    createSuffix="{0}_{1}".format(appConfig.domain, defaultSchemaVersion)
        )
        if mode != AppMode.Evaluate:
            appConfig.create = not(appConfig.resume)

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
        suffixRegex="",
        createSuffix=None):
    """
    Finds all run folders inside the parent folder passed.
    if runIndex is passed, then returns the folder with specified run index.
    if runIndex is not passed, then returns the folder with latest run index.
    """
    existingRunFolders = list(glob.glob(dataFolderPath + allRunsFolder + "run." + "[0-9]"*5 + ".*"))
    existingRunFolders = [fullFolderPath for fullFolderPath in existingRunFolders if os.listdir(fullFolderPath)]
    existingRunFolders = [fullFolderPath[len(dataFolderPath):] for fullFolderPath in existingRunFolders]
    if runIndex is None:
        folderRegex = "runFolders/run\.\d*\." + suffixRegex + "$"
    else:
        folderRegex = "runFolders/run\.0*{0}\.".format(runIndex) + suffixRegex + "$"
    compatibleRunFolders = [runFolder for runFolder in existingRunFolders
                            if re.match(folderRegex, runFolder) ]
    compatibleIndexFolderMap = {}
    for runFolder in compatibleRunFolders:
        runIndex = int(os.path.basename(runFolder)[4:9])
        runFolder = runFolder.replace("\\", "/")+"/"
        curBestFolder = compatibleIndexFolderMap.get(runIndex, None)
        if curBestFolder is None or curBestFolder < runFolder:
            compatibleIndexFolderMap[runIndex] = runFolder
            
    if runIndex is None:
        # Pick a runIndex that we can use.
        if resume in [True, None]:
            # Pick the latest compatible resume-able run.
            if compatibleIndexFolderMap:
               runIndex = max(compatibleIndexFolderMap.keys())
        elif compatibleIndexFolderMap:
            # Pick the next run index.
            runIndex = max(compatibleIndexFolderMap.keys())+1
        else:
            # First run index
            runIndex = 0

    if runIndex is not None:
        # We have a run index to try.
        if runIndex not in compatibleIndexFolderMap.keys():
            # Run index does not currently exist.
            if create in [True, None]:
                runFolder = "{0}run.{1:0>5}.{2}/".format(allRunsFolder, runIndex, createSuffix)
                os.makedirs(dataFolderPath + runFolder, exist_ok=True)
                return runFolder, runIndex, False
            else:
                raise FileNotFoundError("Run folder with specified index doesn't exist.")
        else:
            if runIndex in compatibleIndexFolderMap.keys():
                return compatibleIndexFolderMap[runIndex], runIndex, True
            else:
                raise ValueError("Specified run index incompatible with specified suffix.")
    else:
        if create in [None, True]:
            runIndex = max(compatibleIndexFolderMap.keys()) + 1 if compatibleIndexFolderMap else 0
            runFolder = "{0}run.{1:0>5}.{2}/".format(allRunsFolder, runIndex, createSuffix)
            os.makedirs(dataFolderPath + runFolder, exist_ok=True)
            return runFolder, runIndex, False
        else:
            raise FileNotFoundError("No run exists. To create a new run, set create=True.")

def getLatestCheckpoint(dataFolderPath, runFolder):
    """
    Identifies the latest model weights checkpoint that we can use to resume an interrupted
    indexing operation.
    """
    existingCheckpoints = list(glob.glob(dataFolderPath + runFolder + "Chk*/model.pt"))
    if existingCheckpoints:
        latestCheckpoint = max(existingCheckpoints)
        checkpointFolder = os.path.basename(os.path.dirname(latestCheckpoint))
        stepDotEpochStr = checkpointFolder[len("Chk"):]
        checkpointEpoch, checkpointStep = [int(item) for item in stepDotEpochStr.split(".")]
        return checkpointEpoch, checkpointStep, checkpointFolder
    else:
        return -1, -1, None


