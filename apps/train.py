"""
Entry point for the application which trains the model implementation on any specified domain.
Training progress is rendered on the command line.

Don't call directly. Use ./scripts/train.sh --domain <domainId>

To see command line options, run ./scripts/train.sh --help
"""
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
from hier2hier.dataset import SourceField, TargetField, Hier2HierDataset
from hier2hier.evaluator import Predictor
from hier2hier.util.checkpoint import Checkpoint

from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

# Obtain app configuration object.
appConfig, modelArgs = loadConfig(AppMode.Train)

# Setup logging
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
runFolder = appConfig.training_dir + appConfig.runFolder
os.makedirs(runFolder, exist_ok=True)
logging.basicConfig(
        filename=runFolder + "training.log",
        format=LOG_FORMAT,
        level=getattr(logging, appConfig.log_level.upper()))

# Log config info.
logging.info("Application Config: {0}".format(json.dumps(vars(appConfig), indent=2)))
logging.info("Unprocessed Model Arguments: {0}".format(json.dumps(modelArgs, indent=2)))

# Pick the device, preferably GPU where we run our application.
device = torch.device("cuda") if torch.cuda.is_available() else None

# Trainer object is requred to
trainer = SupervisedTrainer(appConfig, modelArgs, device)

# Load training and dev dataset.
training_data = Hier2HierDataset(baseFolder=appConfig.train_path, fields=trainer.fields, selectPercent=appConfig.input_select_percent)
dev_data = Hier2HierDataset(baseFolder=appConfig.dev_path, fields=trainer.fields, selectPercent=appConfig.input_select_percent)

# Train the model.
trainer.train(training_data, dev_data=dev_data)
