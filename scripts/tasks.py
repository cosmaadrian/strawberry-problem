import random


def task_reverse_the_words_dirty(sentence):
    original = ' '.join(sentence.split(' ')[::-1])
    to_reverse = sentence

    return {
        'task_description': 'Reverse the words',
        'task_token': '<|reverse-the-words|>',
        'param': "",
        'input': original,
        'answer': to_reverse,
    }

def task_reverse_the_words_clean(sentence):
    original = sentence
    to_reverse = ' '.join(sentence.split(' ')[::-1])

    return {
        'task_description': 'Reverse the words',
        'task_token': '<|reverse-the-words|>',
        'param': "",
        'input': original,
        'answer': to_reverse,
    }

def task_reverse_from_dirty(sentence):
    original = sentence[::-1]
    to_reverse = sentence

    return {
        'task_description': 'Reverse',
        'task_token': '<|reverse|>',
        'param': "",
        'input': original,
        'answer': to_reverse,
    }

def task_reverse_from_clean(sentence):
    original = sentence
    to_reverse = sentence[::-1]

    return {
        'task_description': 'Reverse',
        'task_token': '<|reverse|>',
        'param': "",
        'input': original,
        'answer': to_reverse,
    }

def task_reverse_from_dirty_word(sentence):
    words = sentence.split(' ')
    new_sentence = ' '.join([word[::-1] for word in words])

    return {
        'task_description': 'Reverse every word',
        'task_token': '<|reverse-every-word|>',
        'param': "",
        'input': new_sentence,
        'answer': sentence,
    }

def task_reverse_from_clean_word(sentence):
    words = sentence.split(' ')
    new_sentence = ' '.join([word[::-1] for word in words])

    return {
        'task_description': 'Reverse every word',
        'task_token': '<|reverse-every-word|>',
        'param': "",
        'input': sentence,
        'answer': new_sentence,
    }

def task_replace_letters(sentence):
    random_letter = random.choice(list(set(sentence) - set(' !?.,:;')))
    random_letter2 = random.choice(list(set(sentence) - set(random_letter)))

    return {
        'task_description': f'Replace {random_letter2} with {random_letter}',
        'task_token': '<|replace|>',
        'param': (random_letter2, random_letter),
        'input': sentence,
        'answer': sentence.replace(random_letter2, random_letter),
    }

def task_replace_words(sentence):
    words = sentence.split(' ')
    random_word = random.choice(words)
    random_word2 = random.choice(list(set(words) - set(random_word)))

    return {
        'task_description': f'Replace {random_word} with {random_word2}',
        'task_token': '<|replace|>',
        'param': (random_word, random_word2),
        'input': sentence,
        'answer': ' '.join([word if word != random_word else random_word2 for word in words]),
    }

def task_remove_word(sentence):
    words = sentence.split(' ')
    target_word = random.choice(words)

    return {
        'task_description': f'Remove {target_word}',
        'task_token': '<|remove|>',
        'param': target_word,
        'input': sentence,
        'answer': ' '.join([word for word in words if word != target_word]),
    }

def task_remove_word_every_k(sentence):
    words = sentence.split(' ')
    k = random.randint(2, 6)

    return {
        'task_description': f'Remove word every {k}',
        'task_token': '<|remove-word-every|>',
        'param': k,
        'input': sentence,
        'answer': ' '.join([word for i, word in enumerate(words) if (i + 1) % k != 0]),
    }

def task_remove_letter(sentence):
    target_letter = random.choice(list(set(sentence) - set(' !?.,:;')))
    return {
        'task_description': f'Remove {target_letter}',
        'task_token': '<|remove|>',
        'param': target_letter,
        'input': sentence,
        'answer': sentence.replace(target_letter, ''),
    }

def task_remove_letter_every_k(sentence):
    k = random.randint(min(4, len(sentence) // 2), min(6, len(sentence) // 2))

    return {
        'task_description': f'Remove letter every {k}',
        'task_token': '<|remove-letter-every|>',
        'param': k,
        'input': sentence,
        'answer': ''.join([c for i, c in enumerate(sentence) if (i + 1) % k != 0]),
    }

def task_rewrite_with_every_k_words(sentence):
    k = random.randint(1, 4)
    words = sentence.split(' ')
    
    return {
        'task_description': f'Rewrite {k} word',
        'task_token': '<|rewrite-word|>',
        'param': k,
        'input': sentence,
        'answer': ' '.join(words[::k]),
    }

def task_rewrite_uppercase_every_k_words(sentence):
    k = random.randint(1, 4)
    words = sentence.split(' ')
    
    return {
        'task_description': f'Rewrite {k} word upper',
        'task_token': '<|rewrite-word-upper|>',
        'input': sentence,
        'answer': ' '.join([word.upper() if i % k == 0 else word for i, word in enumerate(words)]),
    }

def task_rewrite_with_every_k_letter(sentence):
    k = random.randint(2, min(6, len(sentence) // 2))
    return {
        'task_description': f'Rewrite every {k}',
        'task_token': '<|rewrite-letter|>',
        'input': sentence,
        'answer': sentence[::k],
    }

def task_rewrite_uppercase_every_k_letter(sentence):
    k = random.randint(2, 6)
    return {
        'task_description': f'Rewrite {k} letter upper',
        'task_token': '<|rewrite-letter-upper|>',
        'input': sentence,
        'answer': ''.join([c.upper() if i % k == 0 else c.lower() for i, c in enumerate(sentence)]),
    }

def task_math_round_big(_):
    big_number = random.randint(10000, 100000000)
    position_to_round = random.randint(0, len(str(big_number)) - 1)
    return {
        'task_description': f'Round to {10**position_to_round}',
        'input': str(big_number),
        'answer': str(round(big_number, -position_to_round)),
    }

def task_math_round_small(_):
    small_number = random.uniform(0.00001, 0.9)
    position_to_round = random.randint(1, 5)
    return {
        'task_description': f'Round to {10**-position_to_round}',
        'input': str(small_number),
        'answer': str(round(small_number, position_to_round)),
    }

def task_math_list_parity(_):
    parity = random.choice(['even', 'odd'])
    big_number = random.randint(10000, 100000000)

    return {
        'task_description': f'List {parity} digits',
        'input': str(big_number),
        'answer': ''.join([d for d in str(big_number) if int(d) % 2 == (0 if parity == 'even' else 1)]),
    }

def task_math_comparison_big(_):
    big_number = random.randint(10000, 100000000)
    comparison = random.choice(['>', '<', '==', '>=', '<='])
    comparison_number = big_number + random.randint(-1000, 1000)

    return {
        'task_description': f'',
        'input': f'{big_number} {comparison} {comparison_number}',
        'answer': str(eval(f'{big_number} {comparison} {comparison_number}')),
    }

def task_math_comparison_small(_):
    small_number = random.uniform(0.00001, 0.9)
    comparison = random.choice(['>', '<', '==', '>=', '<='])
    comparison_number = small_number + random.uniform(-0.1, 0.1)

    return {
        'task_description': f'',
        'input': f'{small_number} {comparison} {comparison_number}',
        'answer': str(eval(f'{small_number} {comparison} {comparison_number}')),
    }

def task_count_unigrams(sentence):
    target_character = random.choice(list((sentence)))
    count = sentence.count(target_character)

    return {
        'task_description': f'Count {target_character}',
        'input': sentence,
        'answer': str(count),
    }

def task_count_bigrams(sentence):
    bigrams = list(map(lambda x: ''.join(x), [b for b in zip(sentence[:-1], sentence[1:])]))

    target_bigram = random.choice(bigrams)
    count = sentence.count(target_bigram)

    return {
        'task_description': f'Count {target_bigram}',
        'input': sentence,
        'answer': str(count),
    }

def task_count_trigrams(sentence):
    trigrams = [sentence[i:i+3] for i in range(len(sentence)-2)]
    trigrams = list((trigrams))

    target_trigram = random.choice(trigrams)
    count = sentence.count(target_trigram)

    return {
        'task_description': f'Count {target_trigram}',
        'input': sentence,
        'answer': count,
    }

def task_swap_every_k_words_dirty(sentence):
    k = random.randint(2, 6)

    words = sentence.split(' ')
    
    new_sentence = []
    for i in range(0, len(words), k):
        chunk = words[i: i+k]
        chunk = [chunk[-1]] + chunk[1:-1] + [chunk[0]]
        new_sentence.extend(chunk)

    new_sentence = ' '.join(new_sentence)
    return {
        'task_description': f'Swap every {k} word',
        'task_token': '<|swap-every-word|>',
        'param': k,
        'input': new_sentence,
        'answer': sentence,
    }

def task_swap_every_k_words_clean(sentence):
    k = random.randint(2, 6)

    words = sentence.split(' ')
    
    new_sentence = []
    for i in range(0, len(words), k):
        chunk = words[i: i+k]
        chunk = [chunk[-1]] + chunk[1:-1] + [chunk[0]]
        new_sentence.extend(chunk)

    new_sentence = ' '.join(new_sentence)
    return {
        'task_description': f'Swap every {k} word',
        'task_token': '<|swap-every-word|>',
        'param': k,
        'input': sentence,
        'answer': new_sentence,
    }


def task_swap_every_k_letters_clean(sentence):
    k = random.randint(2, 6)

    new_sentence = ""
    for i in range(0, len(sentence), k):
        chunk = sentence[i: i+k]
        chunk = chunk[-1] + chunk[1:-1] + chunk[0]
        new_sentence += chunk

    return {
        'task_description': f'Swap every {k} letter',
        'task_token': '<|swap-every-letter|>',
        'param': k,
        'input': sentence,
        'answer': new_sentence,
    }

def task_swap_every_k_letters_dirty(sentence):
    k = random.randint(2, 6)

    new_sentence = ""
    for i in range(0, len(sentence), k):
        chunk = sentence[i: i+k]
        chunk = chunk[-1] + chunk[1:-1] + chunk[0]
        new_sentence += chunk

    return {
        'task_description': f'Swap every {k} letter',
        'task_token': '<|swap-every-letter|>',
        'param': k,
        'input': new_sentence,
        'answer': sentence,
    }


TASK_TOKENS = [
    '<|reverse|>',
    '<|replace|>',
    '<|remove|>',
    '<|reverse-the-words|>',
    '<|reverse-every-word|>',
    '<|remove-word-every|>',
    '<|remove-letter-every|>',
    '<|rewrite-word|>',
    '<|rewrite-word-upper|>',
    '<|rewrite-letter|>',
    '<|rewrite-letter-upper|>',
    '<|swap-every-word|>',
    '<|swap-every-letter|>',
]