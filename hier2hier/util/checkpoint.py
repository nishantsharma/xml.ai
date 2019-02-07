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
        tgt_vocab (Vocabulary): vocabulary for the output language

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OBSOLETE_OUTPUT_VOCAB_FILE (str): name of the output vocab file
    """

    CHECKPOINT_DIR_NAME = 'checkpoints/'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    VOCABS_FILE = 'vocabs.pt'

    OBSOLETE_INPUT_VOCABS_FILE = 'input_vocab_{0}.pt'
    OBSOLETE_OUTPUT_VOCAB_FILE = 'tgt_vocab.pt'

    def __init__(self, model, optimizer, loss, epoch, step, batch_size, vocabs):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.vocabs = vocabs
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

        if self.model.schemaVersion == 0:
            for input_vocab_key, input_vocab in self.vocabs.src.items():
                with open(os.path.join(path, self.OBSOLETE_INPUT_VOCABS_FILE.format(input_vocab_key)), 'wb') as fout:
                    dill.dump(input_vocab, fout)
            with open(os.path.join(path, self.OBSOLETE_OUTPUT_VOCAB_FILE), 'wb') as fout:
                dill.dump(self.vocabs.tgt.all, fout)
        else:
            with open(os.path.join(path, self.VOCABS_FILE), 'wb') as fout:
                dill.dump(self.vocabs, fout)

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

        schemaVersion = model.schemaVersion if hasattr(model, "schemaVersion") else 0

        # Extract epoch and step.
        checkpointFolder = os.path.basename(path[0:-1])
        stepDotEpochStr = checkpointFolder[len("Chk"):]
        epoch, step = [int(item) for item in stepDotEpochStr.split(".")]

        # model.flatten_parameters() # make RNN parameters contiguous
        if schemaVersion == 0:
            input_vocab_files = glob.glob(os.path.join(path, cls.OBSOLETE_INPUT_VOCABS_FILE).replace("{0}", "*"))
            src_vocabs = AttrDict()
            for input_vocab_file in input_vocab_files:
                vocab_name = os.path.basename(input_vocab_file)
                start = cls.OBSOLETE_INPUT_VOCABS_FILE.index("{0}")
                end = start + len(vocab_name) - len(cls.OBSOLETE_INPUT_VOCABS_FILE) + 3
                vocab_name  = vocab_name[start:end]
                with open(input_vocab_file, 'rb') as fin:
                    src_vocabs[vocab_name] = dill.load(fin)

            with open(os.path.join(path, cls.OBSOLETE_OUTPUT_VOCAB_FILE), 'rb') as fin:
                tgt_vocab = dill.load(fin)
            vocabs = AttrDict({ "src": src_vocabs, "tgt":{"all":tgt_vocab} })
        else:
            with open(os.path.join(path, cls.VOCABS_FILE), 'rb') as fin:
                vocabs = dill.load(fin)
            
        optimizer = resume_checkpoint['optimizer']
        loss = resume_checkpoint['loss']
        return Checkpoint(model=model,
                        optimizer=optimizer,
                        loss=loss,
                        epoch=epoch,
                        step=step,
                        batch_size=resume_checkpoint.get('batch_size', 100),
                        vocabs=vocabs,
        )
