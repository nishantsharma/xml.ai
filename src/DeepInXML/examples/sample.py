import os
import argparse
import logging
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

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR


def str2bool(v):
    """
    Converts string to bool. This is used as type in argparse arguments to parse
    command line arguments.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', action='store', dest='data_path',
                    help='Path to folder containing dev, train and test input data folders.')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='../../experiment/',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume',
                    default=False, type=str2bool,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument("--save_to_dir",
                    type=str, default="../../output/",
                    help="Target directory where all output is saved.")
parser.add_argument("--debug",
                    type=str2bool, default=False,
                    help="Set to true to enable debug mode.")
parser.add_argument("--profile",
                    type=str2bool, default=False,
                    help="Set to true to enable profiling info printing mode.")

# Build training args needed during training and also inference.
parser.add_argument("--epochs", type = int, default = 400,
                    help="Number of epochs to train for.")
parser.add_argument("--batch_size", type = int, default = 1000,
                    help="Batch size for training.")
parser.add_argument("--num_samples", type = int, default = None,
                    help="Number of samples to train on.")
parser.add_argument("--checkpoint_every", type = int, default = 10,
                    help="Number of epochs after which we take a checkpoint.")
parser.add_argument("--print_every", type = int, default = 10,
                    help="Print progress information, after every so many batches.")

# Schema params.
parser.add_argument("--max_node_count", type = int, default = None,
                    help="Maximum number of nodes in an XML file.")
parser.add_argument("--total_attrs_count", type = int, default = None,
                    help="Total number of known attributes in the schema..")
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
parser.add_argument("--node_text_vec_len", type = int, default = 96,
                    help="Length of encoded vector for node text.")
parser.add_argument("--attrib_value_vec_len", type = int, default = 64,
                    help="Length of an encoded attribute value.")
parser.add_argument("--node_info_propagator_stack_depth", type = int, default = 12,
                    help="Depth of the graph layer stack. This determines the number of "
                    + "hops that information would propagate in the graph inside nodeInfoPropagator.")
parser.add_argument("--propagated_info_len", type = int, default = 64,
                    help="Length of node information vector, when being propagated.")
parser.add_argument("--output_decoder_stack_depth", type = int, default = 7,
                    help="Stack depth of node decoder.")
parser.add_argument("--output_decoder_state_width", type = int, default = 32,
                    help="Width of GRU cell in output decoder.")

# Other meta-parameters of the generated neural network.
parser.add_argument("--input_dropout_p", type = int, default = 0,
                    help="Input dropout probability.")
parser.add_argument("--dropout_p", type = int, default = 0,
                    help="Dropout probability.")
parser.add_argument("--use_attention", type = int, default = True,
                    help="Use attention while selcting most appropriate.")
parser.add_argument("--teacher_forcing_ratio", type = int, default = 0.50,
                    help="Teacher forcing ratio to using during decoder training.")

modelArgs = parser.parse_args()
modelArgs.dev_path = modelArgs.data_path + "dev/"
modelArgs.test_path = modelArgs.data_path + "test/"
modelArgs.train_path = modelArgs.data_path + "train/"

os.makedirs(modelArgs.save_to_dir, exist_ok=True)
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, modelArgs.log_level.upper()))
logging.info(modelArgs)

def updateModelArgsFromData(dataset, modelArgs):
    """
    total_attrs_count, value_symbols_count, max_node_count,
    max_node_fanout, max_node_text_len, max_attrib_value_len,
    max_output_len.
    """
    batch_iterator = torchtext.data.BucketIterator(
        dataset=dataset, batch_size=modelArgs.batch_size,
        sort=False, #sort_within_batch=True,
        #sort_key=lambda x: len(x.src),
        repeat=False)
    batch_generator = batch_iterator.__iter__()


    nodeTagSet = set()
    attrNameSet = set()
    attrValueSymbolSet = set()
    textSymbolSet = set()
    maxFanout = 0
    maxNodeTextLen = 0
    maxAttrValueLen = 0
    maxOutputLen = 0
    for batch in batch_generator:
        # Process input.
        inputTreeList = getattr(batch, hier2hier.src_field_name)
        for inputTree in inputTreeList:
            for node in inputTree.getroot().iter():
                nodeTagSet.add(node.tag)
                for attrName, attrValue in node.attrib.items():
                    attrNameSet.add(attrName)
                    for ch in attrValue:
                        attrValueSymbolSet.add(ch)
                    attrNameSet.add(attrName)
                    maxAttrValueLen = max(maxAttrValueLen, len(attrValue))
                for ch in node.text:
                    textSymbolSet.add(ch)
                maxFanout = max(maxFanout, len(node))
                maxNodeTextLen = max(maxNodeTextLen, len(node.text))

        # Process output.
        outputTextList = getattr(batch, hier2hier.tgt_field_name)
        for outputText in outputTextList:
            maxOutputLen = max(maxOutputLen, len(outputText))

    if modelArgs.num_samples is None:
        modelArgs.num_samples = len(dataset)
    elif modelArgs.num_samples < len(dataset):
        raise ValueError("num_samples smaller than the actual sample count.")

    if modelArgs.max_node_count is None:
        modelArgs.max_node_count = len(nodeTagSet)
    elif modelArgs.max_node_count < len(nodeTagSet):
        raise ValueError("max_node_count smaller than the actual node count.")

    if modelArgs.total_attrs_count is None:
        modelArgs.total_attrs_count = len(attrNameSet)
    elif modelArgs.total_attrs_count < len(attrNameSet):
        raise ValueError("total_attrs_count smaller than the actual attr count.")

    if modelArgs.value_symbols_count is None:
        modelArgs.value_symbols_count = len(attrValueSymbolSet)
    elif modelArgs.value_symbols_count < len(attrValueSymbolSet):
        raise ValueError("Attribute value symbol count set smaller than the actual symbol count.")

    if modelArgs.max_node_fanout is None:
        modelArgs.max_node_fanout = maxFanout
    elif modelArgs.max_node_fanout < maxFanout:
        raise ValueError("max_fanout set smaller than the actual max fanout.")

    if modelArgs.max_node_text_len is None:
        modelArgs.max_node_text_len = maxNodeTextLen
    elif modelArgs.max_node_text_len < maxNodeTextLen:
        raise ValueError("max_node_text_len set smaller than the actual maximum text length.")

    if modelArgs.max_attrib_value_len is None:
        modelArgs.max_attrib_value_len = maxAttrValueLen
    elif modelArgs.max_attrib_value_len < maxAttrValueLen:
        raise ValueError("max_attrib_value_len smaller than the actual maximum attribute value length.")

    if modelArgs.max_output_len is None:
        modelArgs.max_output_len = maxOutputLen
    elif modelArgs.max_output_len < maxOutputLen:
        raise ValueError("maxOutputLen smaller than the actual maximum output length.")

# Pick the device, preferably GPU.
device = torch.device("cuda") if torch.cuda.is_available() else None

if modelArgs.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
           os.path.join(modelArgs.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, modelArgs.load_checkpoint)))
    checkpoint_path = os.path.join(modelArgs.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, modelArgs.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    h2hModel = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    def filterData(example):
        return True

    train = Hier2HierDataset(
        baseFolder=modelArgs.train_path,
        fields = [('src', src), ('tgt', tgt)],
        # transform=transformXml,
        # filter_pred=filterData
    )

    dev = Hier2HierDataset(
        baseFolder=modelArgs.dev_path,
        fields = [('src', src), ('tgt', tgt)],
        # transform=transformXml,
        # filter_pred=filterData
    )


    src.build_vocabs(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # hier2hier.src_field_name = 'src'
    # hier2hier.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab), device=device)
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad, device=device)

    h2hModel = None
    optimizer = None
    if not modelArgs.resume:
        # Initialize model
        updateModelArgsFromData(train, modelArgs)
        h2hModel = Hier2hier(
            modelArgs,
            src.vocabs.tags,
            src.vocabs.text,
            src.vocabs.attribs,
            src.vocabs.attribValues,
            tgt.vocab,
            tgt.sos_id,
            tgt.eos_id,
            device=device
            )

        for param in h2hModel.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(h2hModel.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss,
                        batch_size=modelArgs.batch_size,
                        checkpoint_every=modelArgs.checkpoint_every,
                        print_every=modelArgs.print_every,
                        expt_dir=modelArgs.expt_dir,
                        debug=modelArgs.debug,
                        profile=modelArgs.profile,
                        )

    h2hModel = t.train(h2hModel, modelArgs, train,
                        dev_data=dev,
                        optimizer=optimizer,
                        device=device)

predictor = Predictor(h2hModel, src.vocabs, tgt.vocab)

test_data = Hier2HierDataset(
    baseFolder=modelArgs.test_path,
    fields = [('src', src), ('tgt', tgt)],
)

test_batch_iterator = torchtext.data.BucketIterator(
    dataset=test_data, batch_size=1,
    sort=False, #sort_within_batch=True,
    #sort_key=lambda x: len(x.src),
    device=device, repeat=False)

for i, singleton_batch in enumerate(test_batch_iterator):
    tree_inputs = singleton_batch.src
    tree_inputs = [ ET.tostring(tree_input.getroot()).decode() for tree_input in tree_inputs ]

    try:
        _, predicted_text_outputs = h2hModel(singleton_batch.src)
        predicted_text_outputs = h2hModel.decodeOutput(predicted_text_outputs)
    except ValueError as v:
        predicted_text_outputs = v

    try:
        expected_text_outputs, expected_text_lengths = singleton_batch.tgt
        expected_text_outputs = h2hModel.decodeOutput(expected_text_outputs, expected_text_lengths)
    except ValueError as v:
        expected_text_outputs = v

    print( ("\n"
            + "Iteration {0}\n"
            + "\tTree Input:\t\t{1}\n"
            + "\tPredicted Output:\t{2}\n"
            + "\tExpected Output:\t{3}\n"
        ).format(
            i,
            tree_inputs,
            predicted_text_outputs,
            expected_text_outputs,
        )
    )
