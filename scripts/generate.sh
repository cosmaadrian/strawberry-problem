#!/bin/bash
set -e

python generate_dataset_wikipedia.py --dataset_name wikimedia/wikipedia --subset 20231101.en --split train --output_dir ../assets/ --max_sentences 5000000
python generate_dataset_wikipedia.py --dataset_name wikimedia/wikipedia --subset 20231101.en --split test --output_dir ../assets/ --max_sentences 100


# ##############################################################################################################################
# ##############################################################################################################################

python generate_vocabs.py --num_characters_per_word 4 --vocab_size 256 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 512 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 1024 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 2048 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 4096 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 8192 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 16384 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 32768 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 4 --vocab_size 65536 --output_dir ../assets/tokenizers/

# ##############################################################################################################################
# ##############################################################################################################################

python generate_vocabs.py --num_characters_per_word 6 --vocab_size 256 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 512 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 1024 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 2048 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 4096 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 8192 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 16384 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 32768 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 6 --vocab_size 65536 --output_dir ../assets/tokenizers/

# ##############################################################################################################################
# ##############################################################################################################################

python generate_vocabs.py --num_characters_per_word 8 --vocab_size 256 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 512 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 1024 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 2048 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 4096 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 8192 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 16384 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 32768 --output_dir ../assets/tokenizers/
python generate_vocabs.py --num_characters_per_word 8 --vocab_size 65536 --output_dir ../assets/tokenizers/