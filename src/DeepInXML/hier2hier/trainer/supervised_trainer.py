from __future__ import division
import json, os, random, time, logging, copy

import torch
import torchtext
from torch import optim

import hier2hier
from hier2hier.evaluator import Evaluator
from hier2hier.loss import NLLLoss
from hier2hier.optim import Optimizer
from hier2hier.util import blockProfiler, methodProfiler, lastCallProfile
from hier2hier.util.checkpoint import Checkpoint
from hier2hier.util.tensorboard import TensorBoardHook


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
    def __init__(self, debug, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100,):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

        self.logger = logging.getLogger(__name__)

        self.debug=debug
        self.tensorBoardHook = TensorBoardHook(self.expt_dir, debug.tensorboard)
        self.origValues = None

    @methodProfiler
    def _train_batch(self, input_variable, target_variable, target_lengths, model):
        loss = self.loss

        with blockProfiler("Input Forward Propagation"):
            # Forward propagation
            decoder_outputs, _ = model(input_variable,
                                    target_variable,
                                    target_lengths,
                                    tensorboard_hook=self.tensorBoardHook,
                                    )

        if False:
            oldValues = {}
            for name, param in model.named_parameters():
                oldValues[name] = copy.deepcopy(param.detach().numpy())

            if self.origValues is None:
                self.origValues = oldValues

        with blockProfiler("Batch Loss Calculation"):
            # Get loss
            loss.reset()
            n = target_variable.numel()
            loss.eval_batch(decoder_outputs.view(n, -1), target_variable.view(n,))
        with blockProfiler("Reset model gradient"):
            # Backward propagation
            model.zero_grad()

        with blockProfiler("Loss Propagation Backward"):
            loss.backward()

        with blockProfiler("Step optimizer"):
            self.optimizer.step()

        # simport pdb;pdb.set_trace()

        if False:
            newValues = {}
            for name, param in model.named_parameters():
                newValues[name] = param.detach().numpy()

            for name, param in model.named_parameters():
                print(
                        name, 
                        abs(newValues[name]-oldValues[name]).sum(),
                        abs(newValues[name]-self.origValues[name]).sum()
                    )

        with blockProfiler("Compute loss"):
            return loss.get_loss()

    def _train_epochs(self, data, model, n_epochs, batch_size, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0, device=None):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=batch_size,
            sort=True, sort_within_batch=True,
            sort_key=lambda x: len(x.tgt),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step

        if self.tensorBoardHook is not None:
            self.tensorBoardHook.stepReset(
                step=start_step,
                epoch=start_epoch,
                steps_per_epoch=steps_per_epoch)
        step_elapsed = 0

        for epoch in range(start_epoch, n_epochs + 1):
            self.tensorBoardHook.epochNext()
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            # Partition next batch for iteartion within curent epoch.
            batch_generator = batch_iterator.__iter__()

            # Consuming batches already seen within the checkpoint.
            for n in range(epoch * steps_per_epoch, step):
                log.info("Skipping batch (epoch={0}, step={1}).".format(epoch, n))
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                self.tensorBoardHook.batchNext()
                step += 1
                step_elapsed += 1
                self.tensorBoardHook.batch = step % steps_per_epoch

                input_variables = getattr(batch, hier2hier.src_field_name)
                target_variables, target_lengths = getattr(batch, hier2hier.tgt_field_name)

                loss = self._train_batch(input_variables, target_variables, target_lengths, model)
                if self.debug.profile:
                    print("Profiling info:")
                    print(json.dumps(lastCallProfile(True), indent=2))

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    log_msg = 'Progress: %d%%, Train %s: %.4f/%d=%.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_total,
                        self.print_every,
                        print_loss_avg)
                    log.info(log_msg)
                    self.tensorBoardHook.add_scalar("loss", print_loss_avg)

                    print_loss_total = 0

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step, batch_size=batch_size,
                               input_vocabs=data.fields[hier2hier.src_field_name].vocabs,
                               output_vocab=data.fields[hier2hier.tgt_field_name].vocab).save(self.expt_dir)

            if step == start_step:
                continue
                
            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0

            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data), float('nan')
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, newModelArgs, data,
              dev_data=None,
              optimizer=None, device=None):
        """ Run training for a given model.

        Args:
            model (hier2hier.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (hier2hier.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (hier2hier.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (hier2hier.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
        Returns:
            model (hier2hier.models): trained model.
        """
        log = self.logger

        num_epochs = 5 if newModelArgs is None else newModelArgs.epochs

        # If training is set to resume
        if newModelArgs.resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            model.set_device(device)
            self.optimizer = resume_checkpoint.optimizer

            # Some parameters, loaded from checkpoint, may be overridden in the command line.
            model.max_output_len = newModelArgs.max_output_len
            model.nodeInfoEncoder.max_node_count = newModelArgs.max_node_count
            model.outputDecoder.max_output_len = newModelArgs.max_output_len
            model.outputDecoder.teacher_forcing_ratio = newModelArgs.teacher_forcing_ratio
            model.outputDecoder.runtests = newModelArgs.debug.runtests
            model.debug = newModelArgs.debug

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            if newModelArgs is None:
                batch_size = resume_checkpoint.batch_size
            else:
                batch_size = resume_checkpoint.batch_size
                if newModelArgs.batch_size != resume_checkpoint.batch_size:
                    log.warn("Changing batchsize from {0} to model specified {1}.".format(
                        batch_size,
                        resume_checkpoint.batch_size
                    ))
        else:
            start_epoch = 0
            step = 0
            batch_size = newModelArgs.batch_size
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer


        log.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epochs(data, model, num_epochs, batch_size,
                            start_epoch, step, dev_data=dev_data,
                            device=device)
        return model
