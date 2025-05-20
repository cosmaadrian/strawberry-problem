import torch
import torch.nn as nn

import os
import numpy as np

from acumen_tokenizer import AcumenTokenizer
from transformers import AutoTokenizer

from .building_blocks import TransformerEncoder
from .utils import MuReadout, print_mask

from utils_tokenization import make_mask_by_boundaries, make_block_causal_mask
from utils_tokenization import inter_segment_indices, intra_segment_indices


def prepare_attention_mask(attention_mask):
    attention_mask = attention_mask.float()
    attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    attention_mask[attention_mask == 0] = -10000
    attention_mask[attention_mask == 1] = 0
    return attention_mask

class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.args = args

        from lib import nomenclature
        from lib.accelerator import AcumenAccelerator

        self.nomenclature = nomenclature
        self.accelerator = AcumenAccelerator()

        if 'acumen' in self.args.model_args.input_tokenizer:
            print('::: Loading the ACUMEN TOKENIZER!!!')
            tokenizer = AcumenTokenizer.from_pretrained(args.model_args.input_tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_args.input_tokenizer, token = os.environ.get('HF_TOKEN', None))

        self.token_vocab_size = len(tokenizer)
        self.token_vocab_size += 256 # for start and end answer tokens, empty token, start of text token, and pad token

        if bool(self.args.model_args.use_char_context):
            self.token_dmodel_to_char_dmodel = nn.Linear(
                in_features = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
                out_features = int(self.args.model_args.char_llm_args.dmodel * self.args.model_width_multiplier),
            )

            self.char_transformer = TransformerEncoder(
                args = self.args,
                dmodel = int(self.args.model_args.char_llm_args.dmodel * self.args.model_width_multiplier),
                depth = self.args.model_args.char_llm_args.num_layers,
                nheads = int(8 * self.args.model_width_multiplier),
                dropout = self.args.model_args.char_llm_args.dropout,
                has_context = False,
            )

            self.char_dmodel_to_token_dmodel = nn.Linear(
                in_features = int(self.args.model_args.char_llm_args.dmodel * self.args.model_width_multiplier),
                out_features = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            )

            max_token_size_chars = 32
            
            self.intra_token_pe = nn.Embedding(
                num_embeddings = max_token_size_chars, 
                embedding_dim = self.args.model_args.dmodel,
            )

        ###################################################################
        self.token_correspondence_embedding = nn.Embedding(
            num_embeddings = self.args.dataset_args.chunk_size + 256, # just to be sure idk
            embedding_dim = self.args.model_args.dmodel,
        )

        self.token_embeddings = nn.Embedding(
            num_embeddings = self.token_vocab_size,
            embedding_dim = self.args.model_args.dmodel,
        )
        ###################################################################

        self.model = TransformerEncoder(
            args = self.args,
            dmodel = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            depth = self.args.model_args.num_layers,
            nheads = int(8 * self.args.model_width_multiplier),
            dropout = self.args.model_args.dropout,
            has_context = bool(self.args.model_args.use_char_context),
            context_position = self.args.model_args.char_llm_args.context_position,
        )

        self.decoder_out = MuReadout(
            in_features = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            out_features = self.token_vocab_size,
            args = args,
        )

    def forward(self, batch, **kwargs):
        if bool(self.args.model_args.use_char_context):
            return self.forward_with_context(batch, **kwargs)

        return self.forward_baseline(batch, **kwargs)

    def forward_with_context(self, batch, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        boundaries = batch['boundaries']

        char_input_ids = batch['char_input_ids']
        char_attention_mask = batch.get('char_attention_mask', None)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask)

        if char_attention_mask is not None:
            char_attention_mask = prepare_attention_mask(char_attention_mask)

        #######################################################################
        char_embeddings = self.token_embeddings(char_input_ids)
        indices_per_segment = inter_segment_indices(boundaries) # shape [bs, chunk_size]
        char_embeddings = char_embeddings + self.token_correspondence_embedding(indices_per_segment)
        #################################################################
        ####################### Intra-token PE ##########################

        indices_per_char = intra_segment_indices(boundaries) # shape [bs, chunk_size]

        char_embeddings = char_embeddings + self.intra_token_pe(indices_per_char)
        #################################################################

        #################################################################
        block_causal_mask = make_block_causal_mask(boundaries = boundaries)
        #################################################################

        char_embeddings = self.token_dmodel_to_char_dmodel(char_embeddings)
        char_embeddings = self.char_transformer(char_embeddings, mask = char_attention_mask, causal_mask = block_causal_mask)
        char_embeddings = self.char_dmodel_to_token_dmodel(char_embeddings)

        ########################################################################
        embeddings = self.token_embeddings(input_ids)
        position_indices = torch.arange(0, embeddings.size(1), device = embeddings.device).unsqueeze(0).expand(embeddings.size(0), -1)
        embeddings = embeddings + self.token_correspondence_embedding(position_indices)
        ########################################################################

        ########################################################################
        causal_context_mask = make_mask_by_boundaries(context_len = embeddings.shape[1], boundaries = boundaries)
        ########################################################################

        outputs = self.model(
            embeddings,
            mask = attention_mask,
            causal_mask = True,

            context = char_embeddings,
            context_mask = char_attention_mask,
            causal_context_mask = causal_context_mask,
        )
        out = self.decoder_out(outputs)
        return out

    def forward_baseline(self, batch, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask)

        embeddings = self.token_embeddings(input_ids)

        outputs = self.model(
            embeddings, 
            mask = attention_mask, 
            causal_mask = True, 
        )
        
        out = self.decoder_out(outputs)

        return out
