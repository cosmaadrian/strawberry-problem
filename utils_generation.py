import torch
from utils_tokenization import update_char_input_ids_right_pad, do_tokenize, SpecialTokens

def fast_categorical_sample(logits, temperature=1.0):
    # Add Gumbel noise to logits and take the argmax
    noise = -torch.empty_like(logits).exponential_().log()  # Gumbel(0, 1) noise
    return (logits / temperature + noise).argmax(dim=-1)

def generate(
        model, 
        input_ids, 
        attention_mask, 
        tokenizer, 
        ################################
        char_context = False,
        char_input_ids = None, 
        char_attention_mask = None, 
        boundaries = None, 
        ################################
        max_new_tokens = 32, 
        temperature = 0,
        stop_on_eos = True,
        # not used currently
        min_p = 0.0,
        top_p = 1.0,
    ):

    from utils import catchtime

    end_of_ans_id = tokenizer.convert_tokens_to_ids(SpecialTokens.end_of_answer)
    pad_id = tokenizer.pad_token_id
    
    answer = torch.ones((input_ids.shape[0], max_new_tokens), dtype = torch.long, device = input_ids.device) * pad_id
    attention_mask_answer = torch.zeros((input_ids.shape[0], max_new_tokens), dtype = torch.long, device = input_ids.device)

    input_ids = torch.nn.functional.pad(
        input_ids.detach().clone() ,
        (0, max_new_tokens),  "constant", tokenizer.pad_token_id
    )

    attention_mask = torch.nn.functional.pad(
        attention_mask.detach().clone() ,
        (0, max_new_tokens),  "constant", 0
    )

    is_eos = torch.zeros((input_ids.shape[0], 1), dtype = torch.bool, device = input_ids.device)
    
    for i in range(max_new_tokens):
        logits = model({
                "input_ids": input_ids,
                'attention_mask': attention_mask,
                "char_input_ids": char_input_ids,
                'char_attention_mask': char_attention_mask,
                'boundaries': boundaries,
            }, 
        )

        if temperature == 0:
            token_idx1 = torch.argmax(logits, dim = -1)
        else:
            token_idx1 = fast_categorical_sample(logits, temperature = temperature)

        idxs = attention_mask.sum(dim = 1) - 1
        token_idx = torch.zeros((idxs.shape[0], 1), dtype = torch.long, device = attention_mask.device)
        
        batch_indices = torch.arange(idxs.shape[0], device=token_idx1.device)
        token_idx = token_idx1[batch_indices, idxs]  # Shape: (batch_size, vocab_size) if token_idx1 holds logits
        token_idx = token_idx.unsqueeze(-1)  # Now shape is (batch_size, 1)

        attention_mask_token_ids = torch.ones((attention_mask.shape[0], 1), dtype = torch.long, device = attention_mask.device)

        #####################################################
        batch_indices = torch.arange(idxs.shape[0], device=input_ids.device)
        positions = idxs + 1  # New positions for each sample

        input_ids[batch_indices, positions] = token_idx.squeeze(-1)
        answer[:, i] = token_idx.squeeze(-1)  # Update answer for the current token across all batch entries

        cond = (token_idx.squeeze(-1) == tokenizer.pad_token_id) | (token_idx.squeeze(-1) == tokenizer.eos_token_id)
        new_token_mask = torch.where(cond, torch.tensor(0, device=input_ids.device), torch.tensor(1, device=input_ids.device))

        attention_mask_answer[:, i] = new_token_mask
        attention_mask[batch_indices, positions] = new_token_mask
        #####################################################

        # for every index in token_idx that is an eos token, set is_eos to True
        is_eos = is_eos | (token_idx == tokenizer.eos_token_id) | (token_idx == tokenizer.pad_token_id) | (token_idx == end_of_ans_id)
        
        # for t in tokenizer.batch_decode(input_ids):
        #     print(f"```{t.replace(SpecialTokens.pad, '')}```")
        # input()

        if is_eos.all() and stop_on_eos:
            break

        if not char_context:
            continue

        #######################################################################
        output = do_tokenize(
            tokenizer,
            token_ids = token_idx,
            attention_mask = attention_mask_token_ids,
            char_context = char_context,
        )
        #######################################################################

        char_input_ids, char_attention_mask, boundaries = update_char_input_ids_right_pad(
            char_input_ids,
            char_attention_mask,
            output['char_input_ids'].to(char_attention_mask.device),
            output['char_attention_mask'].to(char_attention_mask.device),
            boundaries.to(char_attention_mask.device),
            pad_token_id = tokenizer.pad_token_id,
        )
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_input_ids': answer,
        'answer_attention_mask': attention_mask_answer,
        'answer_str': tokenizer.batch_decode(answer, skip_special_tokens = False),
    }