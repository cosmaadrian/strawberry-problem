import os
import torch
import time
import json

from .callback import Callback


class ModelCheckpoint(Callback):

    def __init__(self,
            args,
            name = "ModelCheckpoint",
            monitor = 'val_loss',
            direction = 'down',
            dirpath = 'checkpoints/',
            filename="checkpoint",
            save_best_only = True,
            start_counting_at = 0,
            actually_save = True,
        ):

        self.args = args
        self.start_counting_at = start_counting_at
        self.trainer = None
        self.name = name
        self.direction = direction
        self.dirpath = dirpath
        self.filename = filename
        self.save_best_only = save_best_only
        self.actually_save = actually_save

        self.previous_best = None
        self.previous_best_path = None

        self.saved_config = False

        self.monitor = monitor
        self.actual_monitor = 'reconstruction/' + monitor

        if self.args.resume_from != '':
            print('[ModelCheckpoint]: ', self.args.resume_from)
            return

        # check if directory exists and is empty only if we are not overriding checkpoints and we are resuming from a checkpoint
        if not bool(self.args.model_checkpoint.override_checkpoints):
            if os.path.exists(self.dirpath) and os.listdir(self.dirpath):
                raise Exception(f"⚠️ [{name}] Directory {self.dirpath} exists and is not empty.")

    def on_epoch_end(self):
        # use accelerator to get random states
        print(f'[ModelCheckpoint - {self.trainer.accelerator.local_rank}]: Gathering random states.')
        random_states = self.trainer.accelerator.gather_rng_states()

        if self.trainer.accelerator.rank != 0:
            return

        print(f'[ModelCheckpoint - {self.trainer.accelerator.local_rank}]: Gathered random states!')

        if self.trainer.epoch < self.start_counting_at:
            return

        if self.save_best_only:
            if self.actual_monitor not in self.trainer.logger.metrics:
                print(f"⚠️ Metric {self.monitor} not found in logger. Skipping checkpoint.")
                return

            trainer_quantity = self.trainer.logger.metrics[self.actual_monitor]

            if self.previous_best is not None and self.save_best_only:
                if self.direction == 'down':
                    if self.previous_best <= trainer_quantity:
                        print(f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}")
                        return
                else:
                    if self.previous_best >= trainer_quantity:
                        print(f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}")
                        return

            path = os.path.join(self.dirpath, self.filename.format(
                **{'epoch': self.trainer.epoch, self.monitor: trainer_quantity, 'global_step': self.trainer.global_step}
            ))
        else:
            path = os.path.join(self.dirpath, self.filename.format(
                **{'epoch': self.trainer.epoch, 'global_step': self.trainer.global_step}
            ))

        if self.previous_best_path is not None:
            previous_model_path = self.previous_best_path + '.ckpt'

            if self.actually_save:
                if os.path.exists(previous_model_path) and bool(self.args.model_checkpoint.delete_previous):
                    os.unlink(previous_model_path) # TODO don't do this every time, I want to have it as an option
                else:
                    print(f"WARNING: [{self.name}] Previous model path {previous_model_path} not found.")

        if self.actually_save:
            print(f"[{self.name}] Saving model to: {path}")
        else:
            print(f"[{self.name}] (NOT really) Saving model to: {path}")

        os.makedirs(self.dirpath, exist_ok = True)

        self.previous_best = trainer_quantity if self.save_best_only else None
        self.previous_best_path = path

        config_path = os.path.join(self.dirpath, 'config.json')

        if (not os.path.exists(config_path) and not self.saved_config) and self.actually_save:
            with open(config_path, 'wt') as f:
                json.dump(self.args, f, indent = 4)

            self.saved_config = True

        if self.actually_save:
            state_dict = {
                # current iteration
                'current_iter': self.trainer.global_step,
                'current_epoch': self.trainer.epoch,

                # state_dicts
                'model_state_dict': {
                    (k if not k.startswith('module.') else k[len('module.'):]):v
                    for k,v in self.trainer.model_hook.state_dict().items()
                },
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'scheduler_state_dict': self.trainer.scheduler.state_dict(),
                'scaler_state_dict': self.trainer.scaler.state_dict(),

                # random states
                'random_state': random_states,
            }

            torch.save(state_dict, path + '.ckpt')

class TimedCheckpoint(ModelCheckpoint):
        def __init__(self,
                args,
                name = "TimedCheckpoint",
                monitor = 'val_loss',
                direction = 'down',
                dirpath = 'checkpoints/',
                filename="checkpoint",
                save_best_only = True,
                start_counting_at = 0,
                actually_save = True,
                time_interval = 60 * 60,
            ):

            super().__init__(
                args = args,
                name = name,
                monitor = monitor,
                direction = direction,
                dirpath = dirpath,
                filename = filename,
                save_best_only = save_best_only,
                start_counting_at = start_counting_at,
                actually_save = actually_save,
            )

            self.time_interval = time_interval
            self.start_time = time.time()

        def on_batch_end(self):
            # save model every time_interval seconds
            # probably doesn't work with DDP, it results in a deadlock
            if time.time() - self.start_time > self.time_interval:
                print(f"[{self.name}] Saving model. Time interval of {self.time_interval} reached.")
                super().on_epoch_end()
                self.start_time = time.time()

class IterationCheckpoint(ModelCheckpoint):
        def __init__(self,
                args,
                name = "IterationCheckpoint",
                monitor = 'val_loss',
                direction = 'down',
                dirpath = 'checkpoints/',
                filename="checkpoint",
                save_best_only = True,
                start_counting_at = 0,
                actually_save = True,
                interval = 1024,
            ):

            super().__init__(
                args = args,
                name = name,
                monitor = monitor,
                direction = direction,
                dirpath = dirpath,
                filename = filename,
                save_best_only = save_best_only,
                start_counting_at = start_counting_at,
                actually_save = actually_save,
            )

            self.interval = interval

        def on_batch_end(self):
            if (self.trainer.global_step + 1) % self.interval == 0:
                print(f"[{self.name}] Saving model. Iteration interval of {self.interval} reached.")
                super().on_epoch_end()