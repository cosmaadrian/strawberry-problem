import torch
import re

class SpecialTokens:
    start_of_text = '<|start_of_text|>'
    pad = '<|pad|>'
    
    start_of_input = '<|start_of_input|>'
    end_of_input = '<|end_of_input|>'

    start_of_answer = '<|start_of_answer|>'
    end_of_answer = '<|end_of_answer|>'

    char_tokenization = '<|bytes|>'
    end_of_bytes = '<|end_of_bytes|>'
    
    normal_tokenization = '<|toks|>'
    end_of_toks = '<|end_of_toks|>'

    empty = '<|empty|>'

    @classmethod
    def all(cls):
        return [
            cls.start_of_input,
            cls.end_of_input,
            cls.start_of_answer,
            cls.end_of_answer,
            cls.char_tokenization,
            cls.normal_tokenization,
            cls.end_of_bytes,
            cls.end_of_toks,
            cls.empty,
        ]

@torch.compiler.disable
def inter_segment_indices(boundaries):
    segment_indices = torch.cat([
        torch.zeros((boundaries.size(0), 1), device = boundaries.device),
        torch.cumsum(boundaries, dim = 1)
    ], dim = 1)

    segment_indices = segment_indices[:, :-1].long()
    return segment_indices

@torch.compiler.disable
def intra_segment_indices(boundaries):
    batch_size, seq_len = boundaries.size()
    device = boundaries.device
    idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    
    previous_boundary = torch.where(
        boundaries[:, :-1] == 1, 
        torch.arange(seq_len - 1, device=device).unsqueeze(0).expand(batch_size, seq_len - 1),
        torch.full((batch_size, seq_len - 1), -1, dtype=torch.int64, device=device)
    )

    padded = torch.cat([torch.full((batch_size, 1), -1, dtype=torch.int64, device=device), previous_boundary], dim=1)
    last_boundary = padded.cummax(dim=1).values
    return idx - last_boundary - 1

@torch.compiler.disable
def make_mask_by_boundaries(context_len: int, boundaries: torch.Tensor) -> torch.Tensor:
    hh1 = boundaries.cumsum(1) - boundaries
    counts_list = []
    for i in range(boundaries.shape[0]):
        counts = hh1[i].unique_consecutive(return_counts=True)[1]
        
        if len(counts) >= context_len:
            counts = counts[:context_len]

        counts_list.append(counts)

    padded = torch.nn.utils.rnn.pad_sequence(counts_list, batch_first = True, padding_value = 0.)
    if padded.shape[1] < context_len:
        padded = torch.nn.functional.pad(padded, [0, context_len - padded.shape[1]], mode="constant", value = 0.)

    xs = padded.cumsum(1)
    indices = torch.arange(boundaries.size(1), device = xs.device)[None, None, :]
    mask = (indices >= xs.unsqueeze(-1)).long() * -10000
    mask = mask.unsqueeze(1)
    return mask

@torch.compiler.disable
def make_block_causal_mask(boundaries):
    bs, seq_len = boundaries.shape
    device = boundaries.device

    hh1 = boundaries.cumsum(1) - boundaries

    diff = hh1[:, :-1] != hh1[:, 1:]
    end_mask = torch.cat(
        [diff, torch.ones(bs, 1, device=device, dtype=torch.bool)], dim=1
    )

    idxs = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, seq_len)
    ends = torch.where(end_mask, idxs, torch.full_like(idxs, seq_len))
    ends_rev = torch.flip(ends, dims=[1])
    min_rev, _ = torch.cummin(ends_rev, dim=1)
    next_end = torch.flip(min_rev, dims=[1])
    xs = next_end + 1

    seq_range = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
    mask = (seq_range >= xs.unsqueeze(-1)).long() * -10000
    mask = mask.unsqueeze(1)
    return mask

def shift_right_inputs(output, token_tokenizer, char_context = False):
    labels = output['input_ids']

    input_ids = shift_right(labels, token_tokenizer)
    attention_mask = output['attention_mask']
    
    if not char_context:
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    char_input_ids = shift_right(output['char_input_ids'], token_tokenizer)
    char_attention_mask = output['char_attention_mask']

    # shifting boundaries to the right.
    boundaries = output['boundaries']
    boundaries = torch.cat((torch.ones((boundaries.shape[0], 1), dtype = boundaries.dtype, device = boundaries.device), boundaries[:, :-1]), dim = 1)
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,

        'char_input_ids': char_input_ids,
        'char_attention_mask': char_attention_mask,

        'boundaries': boundaries,
    }

def shift_right(input_ids, tokenizer):
    decoder_start_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def update_char_input_ids_right_pad(char_input_ids, char_attention_mask, char_idxs, attn_msk, boundaries, pad_token_id=0):
    increase_size = ((1 - char_attention_mask).sum(dim=1) - attn_msk.sum(dim=1)).min().item()
    increase_size = max(0, -increase_size)
    
    a2 = torch.nn.functional.pad(char_attention_mask, (0, increase_size), "constant", 0)
    b2 = torch.nn.functional.pad(boundaries, (0, increase_size), "constant", 1)
    c2 = torch.nn.functional.pad(char_input_ids, (0, increase_size), "constant", pad_token_id)
    
    idx_start = a2.sum(dim=1, keepdim=True)
    num_chars = attn_msk.sum(dim=1)
    batch_size = c2.shape[0]
    max_chars = num_chars.max().item()
    
    batch_indices = torch.arange(batch_size, device = char_input_ids.device).unsqueeze(1)
    char_positions = torch.arange(max_chars, device = char_input_ids.device).expand(batch_size, -1)
    
    valid_chars_mask = char_positions < num_chars.unsqueeze(1)
    positions = idx_start + char_positions
    
    valid_char_idxs = char_idxs.masked_select(valid_chars_mask.unsqueeze(-1) if char_idxs.dim() > 2 else valid_chars_mask).view(-1)
    valid_positions = positions.masked_select(valid_chars_mask).view(-1)
    valid_batch_indices = batch_indices.expand_as(char_positions).masked_select(valid_chars_mask).view(-1)
    update_indices = torch.stack([valid_batch_indices, valid_positions], dim=0)
    
    boundary_mask = char_positions < (num_chars.unsqueeze(1) - 1)
    valid_boundary_indices = batch_indices.expand_as(char_positions).masked_select(boundary_mask).view(-1)
    valid_boundary_positions = positions.masked_select(boundary_mask).view(-1)

    c2.index_put_((valid_batch_indices, valid_positions), valid_char_idxs)
    a2.index_put_((valid_batch_indices, valid_positions), torch.ones_like(valid_positions, dtype=a2.dtype))
    b2.index_put_((valid_boundary_indices, valid_boundary_positions), torch.zeros_like(valid_boundary_positions, dtype=b2.dtype))

    return c2, a2, b2

def normal_tokenization(text, token_tokenizer):
    _tok = token_tokenizer([text], add_special_tokens = False)
    return {
        'input_ids': _tok['input_ids'][0],
        'attention_mask': _tok['attention_mask'][0],
    }

def byte_tokenization(text, token_tokenizer):
    text = text.replace(SpecialTokens.char_tokenization, '')
    text = text.replace(SpecialTokens.end_of_bytes, '')

    prepend = [token_tokenizer.convert_tokens_to_ids(SpecialTokens.char_tokenization)]
    append = [token_tokenizer.convert_tokens_to_ids(SpecialTokens.end_of_bytes)]

    input_ids = prepend + [token_tokenizer.encode(c)[0] for c in text] + append
    attention_mask = [1] * (len(prepend) + len(text) + len(append))

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    } 

start_b = re.escape(SpecialTokens.char_tokenization)
end_b = re.escape(SpecialTokens.end_of_bytes)
start_t = re.escape(SpecialTokens.normal_tokenization)
end_t = re.escape(SpecialTokens.end_of_toks)
toktype_regex = re.compile(rf'(?P<b>{start_b}.*{end_b})|(?P<t>{start_t}.*{end_t})')

def handle_inputs(input_dict, token_tokenizer):
    tokenized_input_ids = []
    tokenized_attention_mask = []

    for i, _input in enumerate(input_dict['text']):
        text_pieces = []
        last_end = 0

        for mo in toktype_regex.finditer(_input):
            kind = mo.lastgroup
            value = mo.group()
            start, end = mo.start(), mo.end()

            if start != last_end:
                text_pieces.append((normal_tokenization, _input[last_end:start]))
            
            last_end = end

            if kind == 't':
                text_pieces.append((normal_tokenization, value))
            elif kind == 'b':
                text_pieces.append((byte_tokenization, value))
        
        if last_end == 0:
            text_pieces.append((normal_tokenization, _input))
        elif end != len(_input):
            text_pieces.append((normal_tokenization, _input[end:]))

        tokenized_pieces = [fn(piece, token_tokenizer) for fn, piece in text_pieces]

        input_ids = sum([t['input_ids'] for t in tokenized_pieces], [])
        attention_mask = sum([t['attention_mask'] for t in tokenized_pieces], [])

        tokenized_input_ids.append(input_ids)
        tokenized_attention_mask.append(attention_mask)

    tokenized_input_ids = [torch.tensor(inp) for inp in tokenized_input_ids]
    tokenized_attention_mask = [torch.tensor(inp) for inp in tokenized_attention_mask]

    tokenized = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(
            tokenized_input_ids, 
            batch_first = True, 
            padding_value = token_tokenizer.convert_tokens_to_ids(SpecialTokens.pad),
            padding_side = 'right',
        ),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(
            tokenized_attention_mask, 
            batch_first = True, 
            padding_value = 0,
            padding_side = 'right',
        ),
    }

    assert tokenized['input_ids'].shape[0] == tokenized['attention_mask'].shape[0], (tokenized['input_ids'].shape, tokenized['attention_mask'].shape)
    assert tokenized['input_ids'].shape[1] == tokenized['attention_mask'].shape[1], (tokenized['input_ids'].shape, tokenized['attention_mask'].shape)

    return tokenized

def do_tokenize(
        token_tokenizer, 
        input_dict = None, 
        token_ids = None, 
        attention_mask = None,
        char_context = False,
    ):

    if token_ids is None and input_dict is None:
        raise Exception('What are you doing bro?')

    if token_ids is not None and input_dict is not None:
        raise Exception('Make up your mind.')

    if token_ids is None:
        tokenized = handle_inputs(input_dict, token_tokenizer)
    else:
        tokenized = {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
        }

    if not char_context:
        return tokenized

    boundaries_batch = []
    char_batch = []
    for i, (idxs, attn_mask) in enumerate(zip(tokenized['input_ids'], tokenized['attention_mask'])):
        offset = 0
        boundary_i = []
        char_i = []
        for j, idx in enumerate(idxs[:attn_mask.sum()]):

            x = token_tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens = False)
            
            if x is None:
                print("!!!NEVER!!!", idx.item())
                offset += 1
                boundary_i += [1]
                char_i += [token_tokenizer.convert_tokens_to_ids(SpecialTokens.empty)] # this is awkward ...
                continue
            
            if type(x) == list:
                if len(x) == 0: 
                    # print("!!!NEVER!!!")
                    continue
                x = x[0]
            
            x = x.encode().replace(b'\xe2\x96\x81', b'').replace(b'\xc4\xa0', b' ').decode() # remove the prefix
            
            if 'Acumen' in token_tokenizer.__class__.__name__: _condition = x not in token_tokenizer.special2id.keys()
            else: _condition = x not in SpecialTokens.all() + [SpecialTokens.start_of_text, SpecialTokens.pad]

            if _condition:
                char_toks = [token_tokenizer.convert_tokens_to_ids(b) if b != ' ' else token_tokenizer.encode(b)[0] for b in x]
            else:
                char_toks = token_tokenizer.encode(x, add_special_tokens = False)

            if len(char_toks) == 0:
                char_toks = [token_tokenizer.convert_tokens_to_ids(SpecialTokens.empty)]

            assert len(char_toks) > 0, (x, idx, idxs, attn_mask)                

            offset += (len(char_toks))
            boundary_i += [0] * (len(char_toks) - 1) + [1]
            char_i += char_toks

        boundaries_batch.append(boundary_i)
        char_batch.append(char_i)

    # add padding to the boundaries and char input ids, construct char_attention_mask
    max_length = max(len(x) for x in char_batch)
    char_input_ids = torch.full((len(char_batch), max_length), token_tokenizer.pad_token_id, dtype = torch.long)
    char_attention_mask = torch.zeros((len(char_batch), max_length), dtype = torch.long)
    boundaries = torch.ones((len(boundaries_batch), max_length), dtype = torch.long)

    for i, x in enumerate(char_batch):
        char_input_ids[i, :len(x)] = torch.tensor(x)
        
        if len(x) == 1 and (x[0] == token_tokenizer.pad_token_id or x[0] == token_tokenizer.eos_token_id):
            char_attention_mask[i, 0] = 0
        else:
            char_attention_mask[i, :len(x)] = 1

        boundaries[i, :len(boundaries_batch[i])] = torch.tensor(boundaries_batch[i])

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'char_input_ids': char_input_ids,
        'char_attention_mask': char_attention_mask,
        'boundaries': boundaries,
    }