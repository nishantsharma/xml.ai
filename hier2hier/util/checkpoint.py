from __future__ import print_function
import os, time, shutil, glob, json
from orderedattrdict import AttrDict

import torch
import dill

class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        model (hier2hier): hier2hier model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        input_vocab (Vocabulary): vocabulary for the input language
        output_vocab (Vocabulary): vocabulary for the output language

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OUTPUT_VOCAB_FILE (str): name of the output vocab file
    """

    CHECKPOINT_DIR_NAME = 'checkpoints/'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    INPUT_VOCABS_FILE = 'input_vocab_{0}.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'

    def __init__(self, model, optimizer, loss, epoch, step, batch_size, input_vocabs, output_vocab):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.input_vocabs = input_vocabs
        self.output_vocab = output_vocab
        self.epoch = epoch
        self.step = step
        self.batch_size = batch_size
        self._path = None

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, path):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        self._path = path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'batch_size': self.batch_size,
                    'optimizer': self.optimizer,
                    'loss': self.loss,
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, "modelArgs"), 'w') as fout:
            json.dump(self.model.modelArgs, fout, indent=2)

        for input_vocab_key, input_vocab in self.input_vocabs.items():
            with open(os.path.join(path, self.INPUT_VOCABS_FILE.format(input_vocab_key)), 'wb') as fout:
                dill.dump(input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location=map_location)
        model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=map_location)

        # Extract epoch and step.
        checkpointFolder = os.path.basename(path[0:-1])
        stepDotEpochStr = checkpointFolder[len("Chk"):]
        epoch, step = [int(item) for item in stepDotEpochStr.split(".")]

        # model.flatten_parameters() # make RNN parameters contiguous
        input_vocab_files = glob.glob(os.path.join(path, cls.INPUT_VOCABS_FILE).replace("{0}", "*"))
        input_vocabs = AttrDict()
        for input_vocab_file in input_vocab_files:
            vocab_name = os.path.basename(input_vocab_file)
            start = cls.INPUT_VOCABS_FILE.index("{0}")
            end = start + len(vocab_name) - len(cls.INPUT_VOCABS_FILE) + 3
            vocab_name  = vocab_name[start:end]
            with open(input_vocab_file, 'rb') as fin:
                input_vocabs[vocab_name] = dill.load(fin)
        model.inputVocabs = input_vocabs

        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as fin:
            output_vocab = dill.load(fin)
        model.outputVocab = output_vocab

        optimizer = resume_checkpoint['optimizer']
        loss = resume_checkpoint['loss']
        return Checkpoint(model=model, input_vocabs=input_vocabs,
                          output_vocab=output_vocab,
                          optimizer=optimizer,
                          loss=loss,
                          epoch=epoch,
                          step=step,
                          batch_size=resume_checkpoint.get('batch_size', 100)
        )
