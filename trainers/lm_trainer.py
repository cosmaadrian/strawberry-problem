import glob
import torch
import os
import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from torch.optim import AdamW
from lib.trainer_extra import AcumenTrainer


class LMTrainer(AcumenTrainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        from lib.accelerator import AcumenAccelerator
        self.accelerator = AcumenAccelerator()
        self.next_token_prediction_loss = torch.nn.CrossEntropyLoss()
        self.iter_idx = 0

    def load_model_if_needed(self):
        if 'from_base_model' not in self.args:
            return

        if self.args.from_base_model == '':
            return

        self.accelerator.master_print(f"::: [LMTrainer] Loading model from {self.args.from_base_model}...")
        if self.args.from_base_model.endswith('.ckpt'):
            state_dict = torch.load(self.args.from_base_model, weights_only = False)
        else:
            # assume it's a directory
            files = glob.glob(self.args.from_base_model + '/**/*.ckpt')
            if len(files) == 0:
                raise ValueError(f"::: [LMTrainer] No checkpoint files found in {self.args.from_base_model}")
            
            # sort by date and get the latest
            files.sort(key = lambda x: os.path.getmtime(x))
            state_dict = torch.load(files[-1], weights_only = False)

        # TODO? self.model.load_state_dict({'module.' + k:v for k,v in state_dict['model_state_dict'].items()})
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.accelerator.master_print(f"::: [LMTrainer] Loaded!!")

    def initialize_count_matrix(self):
        # 63 = 26 + 26 + 10 + 1 total characters (a-z, A-Z, 0-9, space)
        self.M = np.ones((self.args.vocab_size, 63), dtype = np.int32) * (-np.inf)

        tokenizer = self.dataset.token_tokenizer

        for token_name, token_id in tokenizer.vocab2id.items():
            if token_id <= 62: continue # it's a character

            # put 0s in self.M[token_id - 63] for corresponding characters
            for c in token_name:
                self.M[token_id - 63, tokenizer.vocab2id[c]] = 0

    def training_start(self):
        self.load_model_if_needed()
        # self.initialize_count_matrix()

    def configure_optimizers(self, lr = 0.1):
        if self._optimizer is not None:
            return self._optimizer

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.optimizer_args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        self._optimizer = AdamW(
            params = optim_groups,
            lr = lr,
            betas = [self.args.optimizer_args.beta1, self.args.optimizer_args.beta2],
            weight_decay = self.args.optimizer_args.weight_decay,
            eps = float(self.args.optimizer_args.eps),
            fused = True,
        )

        return self._optimizer

    def compute_mutual_information(self, M):
        # replace -inf with 0
        M = np.where(M == -np.inf, 0, M)

        M = M + 1e-6  # Avoid division by zero

        total = np.sum(M)
        P_tc = M / total  # Joint P(t, c)

        P_t = np.sum(P_tc, axis=1, keepdims=True)  # P(t)
        P_c = np.sum(P_tc, axis=0, keepdims=True)  # P(c)
        
        mask = P_tc > 0
        MI = np.sum(P_tc[mask] * np.log(P_tc[mask] / (P_t @ P_c)[mask]))

        H_T = -np.sum(P_t * np.log(P_t))
        H_C = -np.sum(P_c * np.log(P_c))
        NMI = MI / min(H_T, H_C)

        return {
            "mutual_information": MI,
            "normalized_mutual_information": NMI,
            "entropy_t": H_T,
            "entropy_c": H_C
        }

    def training_batch_start(self, batch = None):
        pass
        # if batch is None:
        #     raise ValueError("Batch is None")

        # for _id_line in batch['input_ids']:
        #     ids_non_char_tokens = _id_line[(_id_line >= 63) & (_id_line < self.args.vocab_size + 63)].detach().cpu().numpy() - 63
        #     ids_char_tokens = _id_line[_id_line < 63].detach().cpu().numpy()

        #     chars, char_counts = np.unique(ids_char_tokens, return_counts = True)
        #     words, word_counts = np.unique(ids_non_char_tokens, return_counts = True)

        #     # update self.M
        #     self.M[np.ix_(words, chars)] += char_counts[None, :]

        # mutual_info = self.compute_mutual_information(self.M.copy())

        # self.log('train/mi', mutual_info['mutual_information'], on_step = False)
        # self.log('train/nmi', mutual_info['normalized_mutual_information'], on_step = False)
        # self.log('train/ht', mutual_info['entropy_t'], on_step = False)
        # self.log('train/hc', mutual_info['entropy_c'], on_step = False)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # print("Trainer: ")
        # for i, t in enumerate(self.dataset.token_tokenizer.batch_decode(input_ids)):
        #     print(batch['text'][i])            
        #     print(t.replace('<|pad|>', ''))
        #     print(input_ids[i])
        # input()

        model_output = self.model({
            "input_ids": input_ids,
            'attention_mask': batch['attention_mask'],

            "char_input_ids": batch.get("char_input_ids", None),
            'char_attention_mask': batch.get('char_attention_mask', None),

            'boundaries': batch.get('boundaries', None),
        })

        # apply the loss only for the answer, not the task description / input.
        next_token_loss = self.next_token_prediction_loss(model_output.view(-1, model_output.size(-1)), labels.view(-1))

        self.log('train/loss:next_byte', next_token_loss.item(), on_step = True)
        self.iter_idx += 1

        # log num tokens seen
        num_tokens = batch['attention_mask'].sum()
        self.log('train/num_tokens', num_tokens.item(), on_step = False, cummulative = True)

        return next_token_loss