import torch
import torch.distributed as dist
from collections import defaultdict

import pandas as pd
pd.set_option('display.max_colwidth', None)
import tqdm
import re
import copy
import time

from lib.evaluator_extra import Metric
from lib.evaluator_extra.acumen_evaluator import AcumenEvaluator
from lib.accelerator import AcumenAccelerator

from utils_tokenization import SpecialTokens
from utils_generation import generate

class InstructionsEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super().__init__(args, model, evaluator_args, logger)
        from lib import nomenclature
        from lib import device
        from models.utils import print_padding_mask

        self.print_mask = print_padding_mask
        self.device = device

        self.efficient_mode = evaluator_args.get('efficient_mode', False)

        arg_copy = copy.deepcopy(args)
        arg_copy.input_tokenization_type = evaluator_args.get('input_tokenization_type', None)
        arg_copy.answer_tokenization_type = evaluator_args.get('answer_tokenization_type', None)
        self.dataset_loader = nomenclature.DATASETS[args.dataset].val_dataloader(arg_copy)
        self.dataset = self.dataset_loader.dataset

        self.extract_ans = re.compile(rf'(.+?){re.escape(SpecialTokens.end_of_answer)}')

        print(f"::: [{self.display_name} - {AcumenAccelerator().local_rank}] Started :::")
        print(f"::: [{self.display_name} - {AcumenAccelerator().local_rank}] Dataset: {args.dataset_args.dataset_name} ({len(self.dataset_loader)} batches) :::")

    @property
    def display_name(self):
        return 'instructions' if 'display_name' not in self.evaluator_args else self.evaluator_args.display_name

    @torch.no_grad()
    def trainer_evaluate(self, global_step = -1):
        counter = 0
        total_loss = 0
        per_token_accuracy = 0
        num_examples = 0
        num_batches = 0
        
        results_dict = defaultdict(list)

        start_time = time.time()
        for batch_idx, batch in tqdm.tqdm(enumerate(self.dataset_loader), total = len(self.dataset_loader), desc = f"[{self.display_name} - {AcumenAccelerator().local_rank}] Evaluating...", dynamic_ncols = True):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            if not self.efficient_mode:
                ##############################################################
                logits = self.model({
                    "input_ids": batch["ft:input_ids"],
                    'attention_mask': batch['ft:attention_mask'],
                    "char_input_ids": batch.get("ft:char_input_ids", None),
                    'char_attention_mask': batch.get('ft:char_attention_mask', None),
                    'boundaries': batch.get('ft:boundaries', None),
                })

                next_latent_token_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["ft:labels"].view(-1), reduction = 'sum')
                total_loss += next_latent_token_loss.item()
                per_token_accuracy += (torch.argmax(logits, dim = -1).view(-1, 1) == batch["ft:labels"].view(-1, 1)).float().mean(axis = 1).mean(axis = 0).item()
                num_examples += (batch["ft:input_ids"].size(0) * batch["ft:input_ids"].size(1))
                num_batches += 1
                ##################################
            
            input_ids = batch["input_ids"]
            attention_mask = batch['attention_mask']
            char_input_ids = batch.get("char_input_ids", None)
            char_attention_mask = batch.get("char_attention_mask", None)
            boundaries = batch.get('boundaries', None)
            
            correct_answers = batch['answer']

            max_new_tokens = max([len(x) for x in correct_answers])

            answers = generate(
                self.model,
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                max_new_tokens = max_new_tokens, 
                tokenizer = self.dataset.token_tokenizer, 
                temperature = 0.0,
                stop_on_eos = True,
                ##########################################
                char_context = self.args.model_args.use_char_context,
                char_input_ids = char_input_ids, 
                char_attention_mask = char_attention_mask, 
                boundaries = boundaries, 
            )

            ###############################################################################################
            
            for xxx, (ans, correct_ans, task_name) in enumerate(zip(answers['answer_str'], correct_answers, batch['task_name'])):
                properly_extracted = False
                try:
                    extracted_answer = self.extract_ans.search(ans).group(1)
                    properly_extracted = True
                except:
                    extracted_answer = ans

                try:
                    correct_ans = self.extract_ans.search(correct_ans).group(1)
                except: pass

                extracted_answer = extracted_answer.replace(SpecialTokens.start_of_answer, '')
                extracted_answer = extracted_answer.replace(SpecialTokens.end_of_answer, '')
                extracted_answer = extracted_answer.replace(SpecialTokens.end_of_bytes, '')
                extracted_answer = extracted_answer.replace(SpecialTokens.end_of_toks, '')
                extracted_answer = extracted_answer.replace(SpecialTokens.pad, '')

                extracted_answer = extracted_answer.strip()
                correct_ans = correct_ans.strip()

                is_correct = extracted_answer == correct_ans

                results_dict['task_name'].append(task_name)
                results_dict['is_correct'].append(int(is_correct))
                results_dict['extracted'].append(int(properly_extracted))
                results_dict['ans_len'].append(len(extracted_answer))
                results_dict['correct_ans_len'].append(len(correct_ans))
                results_dict['answer'].append(extracted_answer)
                results_dict['correct_answer'].append(correct_ans)

        end_time = time.time()

        if AcumenAccelerator().is_distributed:
            all_results = [None] * AcumenAccelerator().world_size   # the container of gathered objects.
            dist.gather_object(obj = pd.DataFrame(results_dict), object_gather_list = all_results)
            all_results = pd.concat(all_results, ignore_index = True)

            counter = torch.tensor(counter, device = self.device)
            dist.all_reduce(counter, op = dist.ReduceOp.SUM)
            counter = counter.item()

            if not self.efficient_mode:
                num_examples = torch.tensor(num_examples, device = self.device)
                dist.all_reduce(num_examples, op = dist.ReduceOp.SUM)
                num_examples = num_examples.item()

                total_loss = torch.tensor(total_loss, device = self.device)
                dist.all_reduce(total_loss, op = dist.ReduceOp.SUM)
                total_loss = total_loss.item()

                per_token_accuracy = torch.tensor(per_token_accuracy, device = self.device)
                dist.all_reduce(per_token_accuracy, op = dist.ReduceOp.SUM)
                per_token_accuracy = per_token_accuracy.item()
            
            num_batches = torch.tensor(num_batches, device = self.device)
            dist.all_reduce(num_batches, op = dist.ReduceOp.SUM)
            num_batches = num_batches.item()

        else:
            all_results = pd.DataFrame(results_dict)

        if not AcumenAccelerator().is_master:
            return []

        # compute accuracy per task on all_results
        task_accuracies = []
        for task_name in all_results['task_name'].unique():
            task_results = all_results[all_results['task_name'] == task_name]
            task_accuracy = task_results['is_correct'].sum() / len(task_results)
            task_accuracies.append(
                Metric(f'{task_name}_acc', value = task_accuracy, monotonicity = ['instant'], evaluator = self)
            )

        average_accuracy = all_results['is_correct'].sum() / len(all_results)
        percent_extracted = all_results['extracted'].sum() / len(all_results)

        task_accuracies.append(
            Metric('avg_acc', value = average_accuracy, monotonicity = ['instant'], evaluator = self)
        )
        
        task_accuracies.append(
            Metric('perc_extracted', value = percent_extracted, monotonicity = ['instant'], evaluator = self)
        )

        if not self.efficient_mode:
            total_loss = total_loss / num_examples
            task_accuracies.append(
                Metric('loss', value = total_loss, monotonicity = ['instant'], evaluator = self)
            )

            per_token_accuracy = per_token_accuracy / num_batches
            task_accuracies.append(
                Metric('per_token_accuracy', value = per_token_accuracy, monotonicity = ['instant'], evaluator = self)
            )

        for task_name, results in all_results.groupby('task_name'):
            results = results.sample(min(10, len(results)))
            print(f"######## [{self.display_name}] Task: {task_name} ########")
            for i, (is_correct, answer, correct_answer) in enumerate(zip(results['is_correct'], results['answer'], results['correct_answer'])):
                if i >= 10:
                    break
                
                print(f"::: ```{correct_answer}```")
                print(f";;; ```{answer}```")
                print(f"::: Correct? {is_correct} :::")
                print()

        print(f"::: Average Accuracy: {average_accuracy:.4f} :::")
        print(f'::: Evaluation time: {end_time - start_time:.2f} seconds :::')
        
        # compute accuracy per quartile on length
        quartiles = [0] + all_results['correct_ans_len'].quantile([0.25, 0.5, 0.75]).tolist() + [all_results['correct_ans_len'].max()]

        for i in range(1, len(quartiles)):
            quartile_results = all_results[
                (all_results['correct_ans_len'] >= quartiles[i - 1]) & 
                (all_results['correct_ans_len'] < quartiles[i])
            ]
            quartile_accuracy = quartile_results['is_correct'].sum() / len(quartile_results)
            task_accuracies.append(
                Metric(f'q{i}_acc', value = quartile_accuracy, monotonicity = ['instant'], evaluator = self)
            )

        return task_accuracies

    @torch.no_grad()
    def evaluate(self, global_step = -1):
        return self.trainer_evaluate(global_step)