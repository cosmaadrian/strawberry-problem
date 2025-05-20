import os
import json
import tqdm
import random
import argparse

from nltk.tokenize import sent_tokenize

import numpy as np
import datasets as hfds

from tasks import *

def sentence_generator(dataset):
    for example in dataset:
        for sentence in sent_tokenize(example['text']):
            sentence = sentence.strip()
            if '\n' in sentence:
                continue

            if ' ' not in sentence:
                continue

            # if the sentence does not contain at least 70% letters, skip
            if sum([1 for c in sentence if c.isalpha()]) / len(sentence) < 0.7:
                continue

            if len(sentence) < 8:
                continue

            if len(sentence) > 64:
                random_idx = random.randint(0, len(sentence) - 64 - 1)
                sentence = sentence[random_idx:random_idx + 64]
            
            sentence = ' '.join(sentence.split()) # remove extra spaces
            yield sentence

TASKS = [
    task_remove_letter,
    task_remove_letter_every_k,
    task_remove_word,
    task_remove_word_every_k,
    task_replace_letters,
    task_replace_words,
    task_reverse_from_clean,
    task_reverse_from_clean_word,
    task_reverse_from_dirty,
    task_reverse_from_dirty_word,
    task_reverse_the_words_clean,
    task_reverse_the_words_dirty,
    task_rewrite_uppercase_every_k_letter,
    task_rewrite_uppercase_every_k_words,
    task_rewrite_with_every_k_letter,
    task_rewrite_with_every_k_words,
    task_swap_every_k_letters_clean,
    task_swap_every_k_letters_dirty,
    task_swap_every_k_words_clean,
    task_swap_every_k_words_dirty,
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, required = True)
    parser.add_argument('--subset', type = str, required = True)
    parser.add_argument('--split', type = str, required = True)
    parser.add_argument('--max_sentences', type = int, required = False, default = 10000)
    parser.add_argument('--output_dir', type = str, required = True)
    parser.add_argument('--overwrite', type = int, required = False, default = 1)
    args = parser.parse_args()

    dataset = hfds.load_dataset(args.dataset_name, args.subset, split = 'train', token = os.environ.get('HF_TOKEN', None))

    # we take only the last fifth of the dataset for the instructions
    # dataset = dataset.shard(5, 4)

    # use 3/4 of the dataset for training, 1/4 for testing
    if args.split == 'train':
        dataset = hfds.concatenate_datasets([dataset.shard(4, i) for i in range(3)])
    else:
        dataset = dataset.shard(4, 3)

    os.makedirs(args.output_dir, exist_ok = True)
    output_file = os.path.join(args.output_dir, f'{args.dataset_name.replace("/", "-")}_{args.subset}_{args.split}.jsonl')

    if os.path.exists(output_file) and args.overwrite:
        os.remove(output_file)

    elif os.path.exists(output_file):
        raise ValueError(f"Output file already exists: {output_file}")

    sentence_generator = sentence_generator(dataset)

    if args.split == 'train':
        for global_idx in tqdm.tqdm(range(args.max_sentences), dynamic_ncols = True):
            task = random.choice(TASKS)

            if 'math_' in task.__name__: # math tasks do not take a sentence as input
                sentence = None
            else:
                try:
                    sentence = next(sentence_generator)
                except StopIteration:
                    print("Ran out of sentences!!")
                    break

            task_name = task.__name__
            task_name = task_name.replace('task_', '').replace('math_', '')
            task_output = task(sentence)
            task_output['task_name'] = task_name
            task_output['_idx'] = global_idx

            with open(output_file, 'a') as f:
                f.write(json.dumps(task_output, sort_keys = True) + '\n')

        print(f"Finished generating: {output_file}")
    
    elif args.split == 'test':
        for i, task in enumerate(TASKS):
            for global_idx in tqdm.tqdm(range(args.max_sentences), dynamic_ncols = True):

                if 'math_' in task.__name__: # math tasks do not take a sentence as input
                    sentence = None
                else:
                    try:
                        sentence = next(sentence_generator)
                    except StopIteration:
                        print("Ran out of sentences!!")
                        break

                task_name = task.__name__
                task_name = task_name.replace('task_', '').replace('math_', '')
                task_output = task(sentence)
                task_output['task_name'] = task_name
                task_output['_idx'] = global_idx + i * args.max_sentences

                with open(output_file, 'a') as f:
                    f.write(json.dumps(task_output, sort_keys = True) + '\n')

            print(f"Finished generating: {output_file}")