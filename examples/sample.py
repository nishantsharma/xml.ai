import os
import argparse
import logging
from orderedattrdict import AttrDict

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

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

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to training data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

# Build training args needed during training and also inference.
parser.add_argument("--batch_size", type = int, default = 64,
                    help="Batch size for training.")
parser.add_argument("--epochs", type = int, default = 100,
                    help="Number of epochs to train for.")
parser.add_argument("--num_samples", type = int, default = 10000,
                    help="Number of samples to train on.")

# Schema params.
parser.add_argument("--total_attrs_count", type = int, default = 128,
                    help="Total number of known attributes in the schema..")
parser.add_argument("--value_symbols_count", type = int, default = 256,
                    help="Total number of symbols used in attribute value strings.")
parser.add_argument("--max_node_count", type = int, default = 8192,
                    help="Maximum number of nodes in an XML file.")
parser.add_argument("--max_node_name_len", type = int, default = 256,
                    help="Maximum length of name of any node.")
parser.add_argument("--max_node_fanout", type = int, default = 128,
                    help="Maximum connectivity fanout of an XML node.")
parser.add_argument("--max_node_attr_len", type = int, default = 256,
                    help="Maximum length of any attribute value at any node.")
parser.add_argument("--max_node_text_len", type = int, default = 16536,
                    help="Maximum length of text attribute in a node.")
parser.add_argument("--max_attrib_value_length", type = int, default = 128,
                    help="Maximum length of any text attribute value in a node.")
parser.add_argument("--max_output_len", type = int, default = 16536,
                    help="Maximum length of the output file.")


# Size meta-parameters of the generated neural network.
parser.add_argument("--node_text_vec_len", type = int, default = 64,
                    help="Length of encoded vector for node text.")
parser.add_argument("--attrib_value_vec_len", type = int, default = 64,
                    help="Length of an encoded attribute value.")
parser.add_argument("--node_info_propagator_stack_depth", type = int, default = 12,
                    help="Depth of the graph layer stack. This determines the number of "
                    + "hops that information would propagate in the graph inside nodeInfoPropagator.")
parser.add_argument("--propagated_info_len", type = int, default = 64,
                    help="Length of node information vector, when being propagated.")
parser.add_argument("--node_decoder_stack_depth", type = int, default = 6,
                    help="Stack depth of node decoder.")
parser.add_argument("--output_decoder_state_width", type = int, default = 6,
                    help="Width of GRU cell in output decoder.")

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    hier2hier = checkpoint.model
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
        baseFolder=opt.train_path,
        fields = [('src', src), ('tgt', tgt)],
        # transform=transformXml,
        # filter_pred=filterData
    )
    dev = Hier2HierDataset(
        baseFolder=opt.dev_path,
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
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    hier2hier = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hier2hier = Hier2hier(
            opt,
            src.vocabs.tags,
            src.vocabs.text,
            src.vocabs.attribs,
            src.vocabs.attribValues,
            tgt.vocab)

        if torch.cuda.is_available():
            hier2hier.cuda()

        for param in hier2hier.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(hier2hier.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, expt_dir=opt.expt_dir)

    hier2hier = t.train(hier2hier, train,
                      num_epochs=6, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(hier2hier, src.vocabs, tgt.vocab)

while True:
    seq_str = input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
