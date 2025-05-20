import os
import torch

# assert torch.cuda.is_available(), '::: CUDA is not available! Not even attempting to start training! :::'

if os.environ.get('RANK', -1) != -1:
    device = torch.device(f'cuda:{os.environ["LOCAL_RANK"]}')
else:
    device = torch.device('cuda')

from ._nomenclature import NOMENCLATURE as nomenclature
from .trainer import NotALightningTrainer

from .dataset_extra import AcumenDataset
from .evaluator_extra import AcumenEvaluator