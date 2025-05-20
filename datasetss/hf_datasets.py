import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.accelerator import AcumenAccelerator
from lib.dataset_extra import AcumenDataset
from acumen_tokenizer import AcumenTokenizer

from transformers import AutoTokenizer
from datasetss import formatters

import os
import sys
import random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import datasets as hfds

from utils_tokenization import do_tokenize, shift_right_inputs, SpecialTokens

import scripts.tasks as tasks

TASKS = [
    tasks.task_remove_letter,
    tasks.task_remove_letter_every_k,
    tasks.task_remove_word,
    tasks.task_remove_word_every_k,
    tasks.task_replace_letters,
    tasks.task_replace_words,
    tasks.task_reverse_from_clean,
    tasks.task_reverse_from_clean_word,
    tasks.task_reverse_from_dirty,
    tasks.task_reverse_from_dirty_word,
    tasks.task_reverse_the_words_clean,
    tasks.task_reverse_the_words_dirty,
    tasks.task_rewrite_uppercase_every_k_letter,
    tasks.task_rewrite_uppercase_every_k_words,
    tasks.task_rewrite_with_every_k_letter,
    tasks.task_rewrite_with_every_k_words,
    tasks.task_swap_every_k_letters_clean,
    tasks.task_swap_every_k_letters_dirty,
    tasks.task_swap_every_k_words_clean,
    tasks.task_swap_every_k_words_dirty,
]

def _collate(list_of_dicts):
    if not len(list_of_dicts):
        raise Exception("Empty list of dicts.")

    keys = list_of_dicts[0].keys()
    return {
        key: [x[key] for x in list_of_dicts]
        for key in keys
    }

class HFDataset(AcumenDataset):

    def __init__(self, args, kind = 'train'):
        super().__init__(args)

        assert os.environ.get('HF_TOKEN', None) is not None, "Please set the HF_TOKEN environment variable to your Huggingface API token"

        self.args = args
        self.kind = kind

        self.chunk_size = args.dataset_args.chunk_size
        self.subset = None if not len(args.dataset_args.subset.replace("'", "").replace('"', "").strip()) else args.dataset_args.subset

        print(self.args.model_args.input_tokenizer)
        ###########################################################
        if 'acumen' in self.args.model_args.input_tokenizer:
            print('::: Loading the ACUMEN TOKENIZER!!!')
            self.token_tokenizer = AcumenTokenizer.from_pretrained(args.model_args.input_tokenizer)
        else:
            self.token_tokenizer = AutoTokenizer.from_pretrained(args.model_args.input_tokenizer, token = os.environ.get('HF_TOKEN', None))
        ###########################################################

        ####################################################################################################################################
        self.token_tokenizer.add_special_tokens({'pad_token': SpecialTokens.pad, 'bos_token': SpecialTokens.start_of_text})
        self.token_tokenizer.add_special_tokens({'additional_special_tokens': SpecialTokens.all() + tasks.TASK_TOKENS})
        ####################################################################################################################################

        # needs to be at the end.
        self.dataset = self.load_dataset()

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, text):
        encoded = text.replace('\n\n', '\n').replace('\n', ' ').encode('utf-8')
        encoded = encoded.strip()
        if self.kind == 'train':
            # randomly truncate text if it's larger than the chunk size
            if len(encoded) > self.chunk_size + 3:
                print(f"Randomly cutting text of length {len(encoded)} to {self.chunk_size}!!!!!!!!!!!!!")
                start = np.random.randint(0, len(encoded) - self.chunk_size)
                encoded = encoded[start:start + self.chunk_size]
        else:
            if len(encoded) > self.chunk_size:
                # truncate text to the chunk size
                print(f"Truncating text of length {len(encoded)} to {self.chunk_size}!!!!!!!!!!!!!")
                encoded = encoded[:self.chunk_size]
            pass

        text = encoded.decode('utf-8', errors = 'ignore')
        return text

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            key: (
                self.preprocess(sample[key]) 
                if key in ['text', 'task_description', 'input', 'answer', 'fulltext']
                else sample[key]
            )
            for key in sample.keys()
        }

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')
        sampler = DistributedSampler(dataset, shuffle = True, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.batch_size,
            pin_memory = True,
            sampler = sampler,
            shuffle = (sampler is None),
            collate_fn = cls.collate_fn(args, dataset.token_tokenizer, kind = 'train'),
            prefetch_factor = 2 if args.environment.extra_args.num_workers > 0 else None,
        )

    @classmethod
    def val_dataloader(cls, args):
        dataset = cls(args = args, kind = 'test')
        sampler = DistributedSampler(dataset, shuffle = False, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.eval_batch_size,
            pin_memory = True,
            shuffle = False,
            sampler = sampler,
            collate_fn = cls.collate_fn(args, dataset.token_tokenizer, kind = 'test'),
            prefetch_factor = 2 if args.environment.extra_args.num_workers > 0 else None,
        )


def make_task_from_tokenizer(tokenizer, target_task = None):
    
    if target_task is None:
        target_task = random.choice(TASKS)

    word_list = [w for w in tokenizer.vocab2id.keys() if len(w) > 2 and '<|' not in w]
    random_sentence = ' '.join([random.choice(word_list) for _ in range(16)])
    task_name = target_task.__name__
    task_name = task_name.replace('task_', '').replace('math_', '')
    task_output = target_task(random_sentence)
    task_output['task_name'] = task_name
    return task_output

class AcumenInstructionDataset(HFDataset):
    class TaskDataset():
        def __init__(self, num_iters, tokenizer): self.num_iters = num_iters; self.tokenizer = tokenizer
        def __len__(self): return self.num_iters
        def __getitem__(self, idx): return make_task_from_tokenizer(self.tokenizer)

    def load_dataset(self):
        print("[AcumenInstructionDataset] Loading dataset", self.args.dataset_args.dataset_name, "subset", self.subset, "split", 'train' if self.kind == 'train' else 'test')
        if self.kind == 'test':
            # 100 samples per task
            total_samples = 100 * len(TASKS) 
            dataset = []
            for task in TASKS:
                for i in range(total_samples // len(TASKS)):
                    task_output = make_task_from_tokenizer(self.token_tokenizer, target_task = task)
                    dataset.append(task_output)
        else:
            dataset = AcumenInstructionDataset.TaskDataset(num_iters = self.args.n_train_iters, tokenizer = self.token_tokenizer)
        return dataset

    @classmethod
    def collate_fn(cls, args, token_tokenizer, kind = 'train'):
        def _collate_fn(batch):
            if kind == 'train':
                formatter = formatters.AcumenInstructionFormatterTrain
            else:
                formatter = formatters.AcumenInstructionFormatterTest

            formatted_input = _collate([formatter.apply_formatting(input_dict) for input_dict in batch])
            output = do_tokenize(token_tokenizer,  input_dict = {'text': formatted_input['text']}, char_context = bool(args.model_args.use_char_context))
            ground_truth = shift_right_inputs(output, token_tokenizer, char_context = bool(args.model_args.use_char_context))
            
            if kind == 'test':
                output_ft = do_tokenize(
                    token_tokenizer, 
                    input_dict = {'text': formatted_input['fulltext']},
                    char_context = bool(args.model_args.use_char_context),
                )
                ground_truth_ft = shift_right_inputs(output_ft, token_tokenizer, char_context = bool(args.model_args.use_char_context))
                output_ft = {
                    f'ft:{key}': value 
                    for key, value in ground_truth_ft.items()
                }
            
            return {
                **ground_truth,
                **{'text': formatted_input['text']},
                **(output_ft if kind == 'test' else {}),
                **({
                    'answer': formatted_input['answer'],
                    'fulltext': formatted_input['fulltext'],
                    'task_name': [x['task_name'] for x in batch],
                } if kind == 'test' else {}),
            }

        return _collate_fn

class FileInstructionDataset(AcumenInstructionDataset):
    def load_dataset(self):
        if 'wikipedia' in self.args.dataset_args.dataset_name:
            print("[InstructionDataset] Loading dataset", self.args.dataset_args.dataset_name, "subset", self.subset, "split", 'train' if self.kind == 'train' else 'test')
            file_name = os.path.join('./assets/', f'{self.args.dataset_args.dataset_name.replace("/", "-")}_{self.args.dataset_args.subset}_{self.kind}.jsonl')

        if not os.path.exists(file_name):
            raise ValueError(f"File {file_name} does not exist")

        # dataset = hfds.load_dataset('json', data_files = file_name)['train']
        df = pd.read_json(file_name, lines = True)
        df['param'] = df['param'].astype(str)
        dataset = hfds.Dataset.from_pandas(df)

        return dataset
