import random
import json
import tqdm
import argparse
import os

def get_really_random_word(unique_words_so_far, max_tries = 100, num_characters_per_word = None):
    tries_so_far = 0
    while True:
        if num_characters_per_word is not None:
            length = num_characters_per_word
        else:
            length = random.randint(3, 7)

        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        word = ''.join(random.choice(letters) for i in range(length))

        if word not in unique_words_so_far:
            break
    
        tries_so_far += 1

        if tries_so_far >= max_tries:
            print(f"Could not find a REALLY random word after {max_tries} tries.")
            return None

    return word

def get_word(unique_words_so_far, max_tries = 100, num_characters_per_word = None):
    word =  get_really_random_word(unique_words_so_far, max_tries = 100, num_characters_per_word = num_characters_per_word)

    if word is None:
        return None

    unique_words_so_far.add(word)
    return word

def make_vocab(vocab_size, K):
    vocab = dict()
    offset = 0

    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'

    # adding the bytes and space
    for token_name in letters + numbers + ' ': 
        vocab[token_name] = offset
        offset += 1

    # adding multi-character tokens
    for i in range(vocab_size):
        word = ''.join(random.choice(letters) for _ in range(K))
        vocab[word] = offset
        offset += 1
    
    return vocab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type = str, required = True)
    parser.add_argument('--num_characters_per_word', type = int, required = True, default = 2)
    parser.add_argument('--vocab_size', type = int, required = True, default = 10000)
    parser.add_argument('--include_bigrams', type = int, required = False, default = 1)
    parser.add_argument('--overwrite', type = int, required = False, default = 1)
    args = parser.parse_args()

    vocab = dict()
    offset = 0

    for token_name in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ': # adding the bytes and space
        vocab[token_name] = offset
        offset += 1

    # make bigrams and add them to the vocab
    if args.include_bigrams:
        for a in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for b in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                bigram = a + b
                
                if bigram not in vocab:
                    vocab[bigram] = offset
                    offset += 1

    unique_words_so_far = set(vocab.keys())
    for i in tqdm.tqdm(range(args.vocab_size)):
        word = get_word(unique_words_so_far, num_characters_per_word = args.num_characters_per_word)

        if word is None:
            continue

        vocab[word] = offset
        offset += 1

    os.makedirs(args.output_dir, exist_ok = True)
    print(f"Saving vocab. Size: {len(vocab)}")
    with open(f'{args.output_dir}/acumen-tokenizer-{args.vocab_size}-{args.num_characters_per_word}{"-bigrams" if args.include_bigrams else ""}.json', 'w') as f:
        json.dump(vocab, f, indent = 4)