import os, argparse, logging, json
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
from hier2hier.util import str2bool, computeAccuracy

from apps.config import AppMode, loadConfig, getLatestCheckpoint, getRunFolder

# For usage help, issue with argument --help.

# Obtain app configuration object.
appConfig, modelArgs = loadConfig(AppMode.Evaluate)

# Setup logging
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, appConfig.log_level.upper()))

# Log config info.
logging.info("Application Config: {0}".format(json.dumps(vars(appConfig), indent=2)))
logging.info("Unprocessed Model Arguments: {0}".format(json.dumps(modelArgs, indent=2)))

# Pick the device, preferably GPU where we run our application.
device = torch.device("cuda") if torch.cuda.is_available() else None

# Trainer object is requred to
trainer = SupervisedTrainer(appConfig, modelArgs, device)
trainer.load()

# Load test dataset.
test_dataset = Hier2HierDataset(baseFolder=appConfig.test_path, fields=trainer.fields, selectPercent=appConfig.input_select_percent)

# Batching test inputs into singletons.
test_batch_iterator = torchtext.data.BucketIterator(
    dataset=test_dataset, batch_size=appConfig.batch_size,
    sort=True, sort_within_batch=True,
    sort_key=lambda x: len(x.tgt),
    device=device, repeat=False)

# Get model from the trainer.
h2hModel = trainer.model
h2hModel.eval()

# In a loop, run the trained model over test dataset.
for i, batch in enumerate(test_batch_iterator):
    tree_inputs = batch.src
    tree_inputs = [ ET.tostring(tree_input.getroot()).decode() for tree_input in tree_inputs ]

    try:
        _, predicted_outputs = h2hModel(batch.src, beam_count=appConfig.beam_count)
        predicted_text = trainer.decodeOutput(predicted_outputs)
    except ValueError as v:
        predicted_text = [v for _ in range(appConfig.batch_size)]

    try:
        expected_outputs, expected_lengths = batch.tgt
        expected_text = trainer.decodeOutput(expected_outputs, expected_lengths)
    except ValueError as v:
        expected_text_outputs = [v for _ in range(appConfig.batch_size)]

    accuracy = computeAccuracy(expected_outputs, expected_lengths, predicted_outputs, device=device)
    print("Accuracy for batch {0}:{1}".format(i, accuracy))

    for j in range(appConfig.batch_size):
        print( ("\n"
                + "Iteration {0}.{1}\n"
                + "\tTree Input:\t\t{2}\n"
                + "\tPredicted Output:\t{3}\n"
                + "\tExpected Output:\t{4}\n"
            ).format(
                i,
                j,
                tree_inputs[j],
                predicted_text[j],
                expected_text[j],
            )
        )
