import models
MODELS = {
    'llm': models.TransformerDecoder
}

import datasetss
DATASETS = {
    'pretraining': datasetss.PretrainingDataset,
    'instructions': datasetss.FileInstructionDataset,
    'instructions-acumen': datasetss.AcumenInstructionDataset,
}

import trainers
TRAINERS = {
    'lm_trainer': trainers.LMTrainer
}

import evaluators
EVALUATORS = {
    'instructions': evaluators.InstructionsEvaluator,
}
