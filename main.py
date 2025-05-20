import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(threshold=10_000)

import os, glob

os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

from torchinfo import summary
import numpy as np

from utils import get_cosine_schedule_with_warmup
from utils import compute_and_set_base_shapes

from easydict import EasyDict
from utils import get_cosine_schedule_with_warmup

import lib.callbacks as callbacks
from lib.arg_utils import define_args

from lib import NotALightningTrainer
from lib import nomenclature
from lib.accelerator import AcumenAccelerator

from colorama import init as colorama_init

colorama_init()

accelerator = AcumenAccelerator()
args = define_args(print_fn = accelerator.master_print, verbose = True)

if not args.resume_from == '':
    accelerator.master_print(f'💚 Trying to resume!')
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/{args.resume_from}/*.ckpt'
    checkpoints = glob.glob(checkpoint_path)

    # Get latest one
    checkpoint = sorted(checkpoints, key = lambda f: os.path.getmtime(f))[-1]

    map_location = {'cuda:%d' % 0: 'cuda:%d' % accelerator.local_rank}
    state_dict = torch.load(checkpoint, map_location = map_location)

    accelerator.master_print(f'💚 Read the checkpoint at {args.resume_from} (Current iter: {state_dict["current_iter"]})')

###############################################################################
###############################################################################
if 'acumen' in args.model_args.input_tokenizer:
    args.vocab_size = int(args.model_args.input_tokenizer.split('-')[-2])
    args.chars_per_token = int(args.model_args.input_tokenizer.split('-')[-1].split('.')[0])

args.model_args.char_llm_args.context_position = list(map(int, args.model_args.char_llm_args.context_position))
###############################################################################
###############################################################################

accelerator.set_args(args)

if args.resume_from != '':
    accelerator.set_rng_state(state_dict['random_state'])

accelerator.prepare_loggers()
logger = accelerator.get_logger()

######################################################
######################################################
######################################################

architecture = nomenclature.MODELS[args.model](args)

try:
    accelerator.master_print(summary(architecture, verbose = 0))
except Exception as e:
    accelerator.master_print("::: ⚠️WARNING⚠️ could not create model summary ::: ", e)

# hack to make a base model
arg_copy = EasyDict(vars(args))
arg_copy.model_width_multiplier = 1
base_architecture = nomenclature.MODELS[args.model](arg_copy)

accelerator.master_print(f'🖥️🖥️🖥️ Computing Base Shapes (muP) 🖥️🖥️🖥️\n')
compute_and_set_base_shapes(model = architecture, base = base_architecture)

# remove the base_architecture, only needed the shapes
del base_architecture
del arg_copy

architecture = accelerator.prepare_model(architecture)
train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args)
    for evaluator_args in args.evaluators
]

# Duplicate code, but it's fine for now ...
if args.n_train_iters != -1:
    actual_epochs = int(np.ceil(args.n_train_iters / (len(train_dataloader) * accelerator.world_size))) # IDK??
    actual_n_train_iters = args.n_train_iters
else:
    actual_epochs = args.epochs
    actual_n_train_iters = args.epochs * len(train_dataloader)

checkpoint_callback_iter = callbacks.IterationCheckpoint(
    args = args,
    name = '🔁 Iteration Checkpoint 🔁',
    monitor = args.model_checkpoint.monitor_quantity,
    dirpath = f'{args.model_checkpoint.base_dir}/{args.group}:{args.name}/iter/',
    save_best_only = False,
    direction = None,
    filename = f'epoch={{epoch}}-step={{global_step}}',
    interval = actual_n_train_iters // args.model_checkpoint.total_saves,
)

max_lr = args.scheduler_args.max_lr

# TODO e pe dos formula, se mareste lr cand batch size scade, trebuie invers
if args.scheduler_args.base_batch_size != args.batch_size:
    max_lr = max_lr * (args.scheduler_args.base_batch_size / args.batch_size)
    accelerator.master_print(f"💚 Adjusting max_lr from {args.scheduler_args.max_lr} to {max_lr} for batch size {args.batch_size}")

optimizer = model.configure_optimizers(lr = max_lr)
scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_training_steps = actual_n_train_iters,
    num_warmup_steps = args.scheduler_args.num_warmup_steps,
    last_epoch = -1
)

if args.resume_from != '':
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    if not bool(args.scheduler_args.reset_scheduler):
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

lr_callback = callbacks.LambdaCallback(
    on_batch_end = lambda: scheduler.step()
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: logger.log('lr', scheduler.get_last_lr()[0])
)

if args.debug:
    accelerator.master_print("[🐞DEBUG MODE🐞] Removing ModelCheckpoint ... ")
    checkpoint_callback_iter.actually_save = False
    # checkpoint_callback_best.actually_save = False
else:
    checkpoint_callback_iter.actually_save = bool(args.model_checkpoint.save_model)
    # checkpoint_callback_best.actually_save = bool(args.model_checkpoint.save_model)

callbacks = [
    # checkpoint_callback_best,
    checkpoint_callback_iter,
    lr_callback,
    lr_logger,
]

trainer = NotALightningTrainer(
    args = args,
    callbacks = callbacks,
    accelerator = accelerator,
    logger = logger,
    scheduler = scheduler,
    state_dict = state_dict if args.resume_from != '' else None,
)

torch.backends.cudnn.benchmark = True
try:
    trainer.fit(
        model,
        optimizer,
        train_dataloader,
        evaluators = evaluators
    )
except KeyboardInterrupt:
    accelerator.master_print("::: 🛑🛑🛑 Training Interrupted 🛑🛑🛑 :::")
    accelerator.terminate()
    exit(-1)