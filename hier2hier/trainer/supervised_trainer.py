from __future__ import division
import json, os, random, time, logging, copy

from orderedattrdict import AttrDict as SortedAttrDict
from attrdict import AttrDict

import torch
import torch.nn as nn
import torchtext
from torch import optim

import hier2hier
from hier2hier.dataset import SourceField, TargetField, Hier2HierDataset, Hier2HierIterator, buildVocabs
from hier2hier.dataset import SYM_PAD
from hier2hier.evaluator import Evaluator
from hier2hier.loss import NLLLoss, Perplexity
from hier2hier.optim import Optimizer
from hier2hier.models import Hier2hier
from hier2hier.util import (blockProfiler, methodProfiler, lastCallProfile, computeAccuracy,
                            summarizeLabelNodes)
from hier2hier.util.checkpoint import Checkpoint
from hier2hier.util import TensorBoardHook, nullTensorBoardHook, AppMode, checkNans

defaultSchemaVersion = 0

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (hier2hier.loss.loss.Loss, optional): loss for training, (default: hier2hier.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, appConfig, modelArgs, device, spotlightByFormula=None):
        # Save configuration info.
        self.appConfig = appConfig
        self.modelArgs = modelArgs
        self.device = device

        if not os.path.exists(self.appConfig.training_dir):
            os.makedirs(self.appConfig.training_dir)

        # Save params.
        self.debug = appConfig.debug

        # Logging.
        self.logger = logging.getLogger(__name__)
        if self.debug.tensorboard != 0:
            self.tensorBoardHook = TensorBoardHook(
                self.debug.tensorboard,
                self.appConfig.training_dir + self.appConfig.runFolder)
        else:
            self.tensorBoardHook = nullTensorBoardHook

        # Field objects contain vocabulary information(Vocab objects) which
        # determines how to numericalize textual data.
        src = SourceField()
        tgt = TargetField()
        self.fields = SortedAttrDict({"src": src, "tgt": tgt,})

        self.model = None
        self.optimizer = None
        self.loss = None
        self.spotlightByFormula = spotlightByFormula

    def load(self, training_data=None):
        # Shortcuts.
        log = self.logger
        device = self.device
        optimizer = self.optimizer
        loss = self.loss
        modelArgs = self.modelArgs

        # If training is set to resume
        if not self.appConfig.resume:
            if training_data is None:
                raise ValueError("Cannot create a new instance of trainer without training data.")

            # Build vocabs.
            srcVocabs, tgtVocabs, tgtToSrcVocabMap = buildVocabs(training_data, max_size=50000)
            src, tgt = self.fields.src, self.fields.tgt
            src.setVocabs(srcVocabs)
            tgt.setVocabs(tgtVocabs)

            # Some of the settings in appConfig and modelArgs need to be deduced from
            # training data. Doing it at creation time.
            self.___updateModelArgsFromData(training_data)

            # Batch counters.
            start_epoch = 0
            step = 0
            batch_size = self.appConfig.batch_size

            # Prepare model object.
            model = Hier2hier(
                self.modelArgs,
                self.debug,
                src.vocabs,
                tgt.vocabs,
                tgt.sos_id,
                tgt.eos_id,
                device=device,
                spotlightByFormula=self.spotlightByFormula,
                )

            # Create optimizer.
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters(), lr=self.modelArgs.learning_rate), max_grad_norm=5)

            # Initialize model weights.
            model.reset_parameters(device)
        else:
            checkpointFolder = "{0}{1}{2}/".format(
                self.appConfig.training_dir,
                self.appConfig.runFolder,
                self.appConfig.checkpointFolder,
                )
            resume_checkpoint = Checkpoint.load(checkpointFolder)
            src, tgt = self.fields.src, self.fields.tgt
            src.setVocabs(resume_checkpoint.vocabs.src)
            tgt.setVocabs(resume_checkpoint.vocabs.tgt)

            model = resume_checkpoint.model
            model.set_device(device)
            if device is not None:
                model.cuda()

            # Define next schema
            if self.modelArgs.schemaVersion is not None:
                self.modelArgs.schemaVersion = modelArgs.schemaVersion
            elif hasattr(model, "schemaVersion"):
                self.modelArgs.schemaVersion = model.schemaVersion
            else:
                self.modelArgs.schemaVersion = defaultSchemaVersion

            # Schema migration.
            model.upgradeSchema(self.modelArgs.schemaVersion)

            # Reconfiguraation upon a new launch.
            modelArgs = model.reconfigureUponLoad(self.modelArgs, self.debug)

            # During testing, we disable spotlight by using spotlightByFormula.
            # It is a test thing.
            model.outputDecoder.spotlightByFormula=self.spotlightByFormula

            optimizer = resume_checkpoint.optimizer

            # A work around to set optimizing parameters properly
            resume_optim = optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            if modelArgs.learning_rate is not None:
            	defaults["lr"] = modelArgs.learning_rate
            optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            if (self.appConfig.batch_size != resume_checkpoint.batch_size):
                log.warn("Changing batch_size from cmdline specified {0} to model specified {1}.".format(
                    self.appConfig.batch_size,
                    resume_checkpoint.batch_size
                ))
                batch_size = resume_checkpoint.batch_size
            else:
                batch_size = resume_checkpoint.batch_size

        # Prepare loss object.
        if loss is None:
            weight = torch.ones(len(tgt.vocabs.all), device=device)
            pad = tgt.vocabs.all.stoi[SYM_PAD]
            loss = Perplexity(weight, pad, device=device)

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.step = step
        self.epoch = start_epoch
        self.modelArgs = modelArgs

        log.info("Final model arguments: {0}".format(json.dumps(self.modelArgs, indent=2)))
        log.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

    def train(self, training_data, dev_data=None):
        """ Run training for a given model.

        Args:
            training_dataset (hier2hier.dataset.dataset.Dataset): dataset object to train on
            dev_dataset (hier2hier.dataset.dataset.Dataset, optional): dev Dataset (default None)
        Returns:
            None. The trained model (hier2hier.models) can be accessed using self.model.
        """
        if self.model is None:
            self.load(training_data)

        self.model.train()
        self._train_epochs(training_data, dev_data)
        return self.model

    @methodProfiler
    def decodeOutput(self, textOutputs, textLengths=None):
        decodedOutputs = []
        sos_id = self.fields.tgt.sos_id
        eos_id = self.fields.tgt.eos_id
        outputVocab = self.fields.tgt.vocabs.all
        max_output_len = self.modelArgs.max_output_len

        for index, textOutput in enumerate(textOutputs):
            textLength = textLengths[index] if textLengths is not None else max_output_len
            if textOutput[0] != sos_id:
                raise ValueError("sos_id missing at index 0.")
            if textLengths is not None:
                if textOutput[int(textLengths[index])-1] != eos_id:
                    raise ValueError("eos_id missing at end index {0}.".format(textLengths[index]))

                indexRange = range(1, int(textLength))
            else:
                indexRange = range(1, min(max_output_len, len(textOutput)))

            decodedOutput = ""
            foundEos = False

            for textIndex in indexRange:
                if textOutput[textIndex] == eos_id:
                    foundEos = True
                    break
                decodedOutput += outputVocab.itos[textOutput[textIndex]]

            if not foundEos:
                decodedOutput += "..."
            decodedOutputs.append(decodedOutput)

        return decodedOutputs


    def ___updateModelArgsFromData(self, training_data):
        """
        total_attrs_count, value_symbols_count, max_node_count,
        max_node_fanout, max_node_text_len, max_attrib_value_len,
        max_output_len.
        """
        # Shortcuts.
        appConfig = self.appConfig
        modelArgs = self.modelArgs

        batch_iterator = torchtext.data.BucketIterator(
            dataset=training_data, batch_size=appConfig.batch_size,
            sort=False, sort_within_batch=False,
            repeat=False)
        batch_generator = batch_iterator.__iter__()

        nodeTagSet = set()
        attrNameSet = set()
        attrValueSymbolSet = set()
        textSymbolSet = set()
        maxFanout = 0
        maxNodeCount = 0
        maxNodeTextLen = 0
        maxAttrValueLen = 0
        maxOutputLen = 0
        for batch in batch_generator:
            # Process input.
            inputTreeList = getattr(batch, hier2hier.src_field_name)
            for inputTree in inputTreeList:
                treeNodes = list(inputTree.getroot().iter())
                maxNodeCount = max(len(treeNodes), maxNodeCount)
                for node in treeNodes:
                    nodeTagSet.add(node.tag)
                    for attrName, attrValue in node.attrib.items():
                        attrNameSet.add(attrName)
                        for ch in attrValue:
                            attrValueSymbolSet.add(ch)
                        attrNameSet.add(attrName)
                        maxAttrValueLen = max(maxAttrValueLen, len(attrValue))
                    if node.text is not None:
                        for ch in node.text:
                            textSymbolSet.add(ch)
                        maxNodeTextLen = max(maxNodeTextLen, len(node.text))
                    if node.tail is not None:
                        for ch in node.tail:
                            textSymbolSet.add(ch)
                        maxNodeTextLen = max(maxNodeTextLen, len(node.tail))
                    maxFanout = max(maxFanout, len(node))

            # Process output.
            _, outputLengths = getattr(batch, hier2hier.tgt_field_name)
            for outputLength in outputLengths:
                maxOutputLen = max(maxOutputLen, int(outputLength))

        if modelArgs.max_node_count is None:
            modelArgs.max_node_count = maxNodeCount
        elif modelArgs.max_node_count < maxNodeCount:
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

    def _train_epochs(self, training_data, dev_data):
        log = self.logger
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        epoch_accuracy_total = 0  # Reset every epoch
        epoch_beamAccuracy_total = 0  # Reset every epoch

        batch_iterator = Hier2HierIterator(
            preprocess_batch=self.model.preprocess_batch,
            dataset=training_data, batch_size=self.batch_size,
            sort=False, shuffle=False, sort_within_batch=False,
            sort_key=lambda x: len(x.tgt),
            device=self.device,
            repeat=False,
        )

        steps_per_epoch = len(batch_iterator)
        if self.step not in range(self.epoch*steps_per_epoch, (self.epoch+1)*steps_per_epoch):
            # Step out of range with epoch.
            self.step = self.epoch * steps_per_epoch
        start_step = self.step
        total_steps = steps_per_epoch * self.appConfig.epochs

        self.tensorBoardHook.stepReset(
            step=self.step,
            epoch=self.epoch,
            steps_per_epoch=steps_per_epoch)

        # Evaluator helps monitor results during training iterations.
        if dev_data is not None:
            evaluator = Evaluator(
                self.model, self.device, dev_data, loss=self.loss, batch_size=self.batch_size,
                mode=self.appConfig.mode,
            )

        first_time = True
        while self.epoch != self.appConfig.epochs:
            self.tensorBoardHook.epochNext()
            # Partition next batch for iteartion within curent epoch.
            batch_generator = batch_iterator.__iter__(self.appConfig.mode)

            if first_time:
                first_time = False
                # Consuming batches already seen within the checkpoint.
                for n in range(self.epoch * steps_per_epoch, self.step):
                    log.info("Skipping batch (epoch={0}, step={1}).".format(self.epoch, n))
                    next(batch_generator)

            log.debug("Currently at Epoch: %d, Step: %d" % (self.epoch, self.step))
            print("Currently at Epoch: %d, Step: %d" % (self.epoch, self.step))

            self.model.train(True)
            calcAccuracy=False#(self.epoch % 100 == 0)
            for hier2hierBatch in batch_generator:
                self.step += 1
                self.epoch = int(self.step / steps_per_epoch)

                self.tensorBoardHook.batchNext()
                self.tensorBoardHook.batch = self.step % steps_per_epoch

                loss, accuracy, beamAccuracy = self._train_batch(
                    hier2hierBatch,
                    calcAccuracy=calcAccuracy,
                    )
                if self.debug.profile:
                    print("Profiling info:")
                    print(json.dumps(summarizeLabelNodes(lastCallProfile(True)), indent=2))

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss
                epoch_accuracy_total += accuracy if accuracy is not None else 0
                epoch_beamAccuracy_total += beamAccuracy if beamAccuracy is not None else 0

                if self.step % self.appConfig.print_every == 0:
                    print_loss_avg = print_loss_total / self.appConfig.print_every
                    log_msg = 'Progress: %d%%, Train %s: %.4f/%d=%.4f' % (
                        self.step / total_steps * 100,
                        self.loss.name,
                        print_loss_total,
                        self.appConfig.print_every,
                        print_loss_avg)
                    log.info(log_msg)
                    print(log_msg)

                    print_loss_total = 0

                # Checkpoint
                if ((
                        self.step % self.appConfig.checkpoint_every == 0
                        or self.step == total_steps
                    )
                    and self.appConfig.mode != int(AppMode.Test)
                ):
                    checkpointFolder = "{0}{1}Chk{2:06}.{3:07}".format(
                        self.appConfig.training_dir,
                        self.appConfig.runFolder,
                        self.epoch, # Cur epoch
                        self.step)

                    vocabs = AttrDict({
                        "src" : training_data.fields["src"].vocabs,
                        "tgt" : training_data.fields["tgt"].vocabs,
                    })
                    Checkpoint(model=self.model,
                               optimizer=self.optimizer,
                               loss=self.loss,
                               epoch=self.epoch,
                               step=self.step,
                               batch_size=self.batch_size,
                               vocabs=vocabs,
                            ).save(checkpointFolder)

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, self.step - start_step)
            self.tensorBoardHook.add_scalar("loss", epoch_loss_avg)
            log_msg = "Finished epoch %d: Train %s: %.4f" % (
                self.epoch,
                self.loss.name,
                epoch_loss_avg,
            )
            epoch_loss_total = 0

            if accuracy is not None:
                epoch_accuracy_avg = epoch_accuracy_total / min(steps_per_epoch, self.step - start_step)
                self.tensorBoardHook.add_scalar("accuracy", epoch_accuracy_avg)
                log_msg += ", Accuracy:%.4f" % (epoch_accuracy_avg)
            epoch_accuracy_total = 0

            if beamAccuracy is not None:
                epoch_beamAccuracy_avg = epoch_beamAccuracy_total / min(steps_per_epoch, self.step - start_step)
                self.tensorBoardHook.add_scalar("beamAccuracy", epoch_beamAccuracy_avg)
                log_msg += ", BeamAccuracy:%.4f" % (epoch_beamAccuracy_avg)
            epoch_beamAccuracy_total = 0

            if dev_data is not None:
                dev_loss, dev_accuracy, dev_beamAccuracy = evaluator.evaluate(calcAccuracy)

                self.optimizer.update(dev_loss, self.epoch)
                log_msg += ", Dev Loss: %.4f" % (dev_loss)
                self.tensorBoardHook.add_scalar("dev_loss", dev_loss)

                if dev_accuracy is not None:
                    log_msg += ", Dev Accuracy %.4f" % (dev_accuracy)
                    self.tensorBoardHook.add_scalar("dev_accuracy", dev_accuracy)

                if dev_beamAccuracy is not None:
                    log_msg += ", Dev beamAccuracy %.4f." % (dev_beamAccuracy)
                    self.tensorBoardHook.add_scalar("dev_beamAccuracy", dev_beamAccuracy)

                self.model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, self.epoch)

            log.info(log_msg)
            print(log_msg)

    @methodProfiler
    def _train_batch(self, hier2hierBatch, calcAccuracy=False):
        with blockProfiler("Input Forward Propagation"):
            decodedSymbolsByTdolList, decodedSymbols = self.model(
                                    hier2hierBatch,
                                    self.tensorBoardHook,
                                    collectOutput=calcAccuracy,
                                    debugAttention=self.appConfig.debug.attention,
                                    )
        if calcAccuracy:
            targetOutputsByToi = hier2hierBatch.targetOutputsByToi
            targetOutputLengthsByToi = hier2hierBatch.targetOutputLengthsByToi
            accuracy = computeAccuracy(targetOutputsByToi, targetOutputLengthsByToi, decodedSymbols, device=self.device)
            print("Batch Accuracy {0}".format(accuracy))
            _, beamDecodedSymbols = self.model(
                                hier2hierBatch,
                                beam_count=4,
                                collectOutput=calcAccuracy,
                                tensorBoardHook=self.tensorBoardHook,
                                )
            beamAccuracy = computeAccuracy(targetOutputsByToi, targetOutputLengthsByToi, beamDecodedSymbols, device=self.device)
        else:
            accuracy, beamAccuracy = None, None

        with blockProfiler("Batch Loss Calculation"):
            # Get loss
            self.loss.reset()
            targetOutputsByTdolList = hier2hierBatch.targetOutputsByTdolList
            assert(len(targetOutputsByTdolList) == len(decodedSymbolsByTdolList))
            for i, decodedSymbolsByTdol in enumerate(decodedSymbolsByTdolList):
                self.loss.eval_batch(decodedSymbolsByTdol, targetOutputsByTdolList[i])

        with blockProfiler("Reset model gradient"):
            # Backward propagation
            self.model.zero_grad()

        # Propagate loss backward to update gradients.
        with blockProfiler("Loss Propagation Backward"):
            self.loss.backward()

        # Clip gadients to handle exploding gradients problem.
        if self.modelArgs.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.modelArgs.clip_gradient, norm_type=float('inf'))

        # Log info into tensorboard.
        for name, param in self.model.named_parameters():
            self.tensorBoardHook.add_histogram(name, param)
            if param.grad is not None:
                self.tensorBoardHook.add_histogram(name + "_grad", param.grad)

        with blockProfiler("Step optimizer"):
            self.optimizer.step()

        with blockProfiler("Compute loss"):
            loss = self.loss.get_loss()

        checkNans([loss, accuracy, beamAccuracy])
        return loss, accuracy, beamAccuracy
