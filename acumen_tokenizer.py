import json
import re


class AcumenTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab2id = json.load(open(vocab_file))
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}

        self.max_id = max(self.vocab2id.values()) + 1

        self.eos_token = '<|end_of_text|>'
        self.eos_token_id = self.max_id
        
        self.special2id = {self.eos_token: self.eos_token_id}
        self.id2special = {self.eos_token_id: self.eos_token}

    def __str__(self):
        return f'AcumenTokenizer(\nvocab={vocab2id}\nspecial_tokens={self.special2id})'

    def __len__(self):
        return len(self.vocab2id) + len(self.special2id)

    def repr(self):
        return self.__str__()

    def __call__(self, text, add_special_tokens = True):
        # TODO return_tensors = 'pt', padding = 'longest'
        is_list = True
        if type(text) == str:
            is_list = False
            text = [text]
        
        input_ids = []
        attention_mask = []
        
        for t in text:
            _input_ids = []
            _attention_mask = []
            
            if add_special_tokens and self.bos_token_id is not None:
                _input_ids.append(self.bos_token_id)
                _attention_mask.append(1)

            split_regex = re.compile(fr'(\b|{"|".join(map(re.escape, self.special2id.keys()))})')

            for i, w in enumerate(split_regex.split(t)):
                if w == '':
                    continue

                try:
                    if w in self.vocab2id:
                        _input_ids.append(self.vocab2id[w])
                        _attention_mask.append(1)
                    elif w in self.special2id:
                        _input_ids.append(self.special2id[w])
                        _attention_mask.append(1)
                    else:
                        # it's a word made out of bigrams?
                        for j in range(0, len(w), 2):
                            if w[j:j+2] in self.vocab2id:
                                _input_ids.append(self.vocab2id[w[j:j+2]])
                                _attention_mask.append(1)
                            else:
                                _input_ids.append(self.vocab2id[w[j]])
                                _attention_mask.append(1)

                                if j + 1 < len(w):
                                    _input_ids.append(self.vocab2id[w[j+1]])
                                    _attention_mask.append(1)

                except Exception as e:
                    print(f'[AcumenTokenizer] Error at word `{w}` in text `{t}`')
                    raise e
            
            if add_special_tokens:
                _input_ids.append(self.eos_token_id)
                _attention_mask.append(1)

            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
        
        return {
            'input_ids': input_ids if is_list else input_ids[0], 
            'attention_mask': attention_mask if is_list else attention_mask[0],
        }
    
    @staticmethod
    def from_pretrained(path, **kwargs):
        return AcumenTokenizer(path, **kwargs)
    
    def add_special_tokens(self, special_tokens_dict):
        if 'pad_token' in special_tokens_dict:
            self.max_id += 1
            self.pad_token_id = self.max_id
            self.pad_token = special_tokens_dict['pad_token']

            self.special2id[special_tokens_dict['pad_token']] = self.pad_token_id
            self.id2special[self.pad_token_id] = special_tokens_dict['pad_token']
        
        if 'bos_token' in special_tokens_dict:
            self.max_id += 1
            self.bos_token_id = self.max_id
            self.bos_token = special_tokens_dict['bos_token']

            self.special2id[special_tokens_dict['bos_token']] = self.bos_token_id
            self.id2special[self.bos_token_id] = special_tokens_dict['bos_token']

        if 'additional_special_tokens' in special_tokens_dict:
            for token in special_tokens_dict['additional_special_tokens']:
                self.max_id += 1
                self.special2id[token] = self.max_id
                self.id2special[self.max_id] = token

    def convert_ids_to_tokens(self, ids, skip_special_tokens = False):
        is_list = True
        
        if type(ids) == int:
            is_list = False
            ids = [ids]

        if type(ids) != list:
            ids = ids.detach().cpu().tolist()

        tokens = []
        for _id in ids:
            if skip_special_tokens and _id in self.id2special:
                continue
            
            if _id in self.id2special:
                tokens.append(self.id2special[_id])
            elif _id in self.id2vocab:
                tokens.append(self.id2vocab[_id])
            else: 
                # print(f'[AcumenTokenizer] ID {_id} not found in vocab or special tokens.', _id, min(self.id2vocab.keys()), max(self.id2vocab.keys()), self.id2special.keys())
                pass
        
        if not is_list and len(tokens) == 1:
            return tokens[0]

        return tokens

    def convert_tokens_to_ids(self, tokens, add_special_tokens = False):
        output = self(tokens, add_special_tokens = add_special_tokens)
        ids = output['input_ids']

        if len(ids) == 1:
            return ids[0]

        return ids

    def encode(self, ids, add_special_tokens = False):
        if type(ids) == str:
            ids = [ids]

        return self.convert_tokens_to_ids(ids, add_special_tokens = add_special_tokens)

    def decode(self, ids, skip_special_tokens = False):
        return ''.join(self.convert_ids_to_tokens(ids, skip_special_tokens))
    
    def batch_decode(self, ids, skip_special_tokens = False):
        return [
            self.decode(_ids, skip_special_tokens) for _ids in ids
        ]

if __name__ == '__main__':
    from utils_tokenization import SpecialTokens
    from scripts.tasks import TASK_TOKENS

    text = '<|remove-word-every|>mama<|reverse|> are <|reverse|>    mere ana are pere  '
    # tok = AcumenTokenizer.from_pretrained('./assets/tokenizers/acumen-tokenizer-8192-6.json')
    tok = AcumenTokenizer.from_pretrained('./assets/tokenizers/acumen-tokenizer-8192-6-bigrams.json')
    
    tok.add_special_tokens({'pad_token': SpecialTokens.pad, 'bos_token': SpecialTokens.start_of_text})
    tok.add_special_tokens({'additional_special_tokens': SpecialTokens.all() + TASK_TOKENS})

    print(f"```{text}```")
    
    input_ids = tok([text], add_special_tokens = True)['input_ids']
    print(input_ids)
    
    decoded = tok.batch_decode(input_ids)
    print(decoded)

    reencoded = tok(decoded, add_special_tokens = False)
    print(reencoded['input_ids'])

    print(tok.batch_decode(reencoded['input_ids']))