#!/bin/bash
export HF_TOKEN=`cat ./.hf_token`

cd ..
set -e

# Training on Wikipedia
##############################################
##############################################
STUFF="--config_file configs/instructions-wikipedia.yaml --mode run --group wiki --eval_every_batches 4096"
python main.py \
        --name base-wiki-gpt2 \
        --model_args.input_tokenizer openai-community/gpt2 \
        --dataset_args.chunk_size 512 \
        --eval_batch_size 512\
        --batch_size 64 \
        --model_args.use_char_context 0 \
        --model_checkpoint.save_model 1\
        --actually_stop_at 750000 \
        --n_train_iters 750000 \
        --debug 0 \
        $STUFF

STUFF="--config_file configs/instructions-wikipedia.yaml --mode run --group wiki --eval_every_batches 4096"
python main.py \
        --name ours-wiki-gpt2 \
        --model_args.input_tokenizer openai-community/gpt2 \
        --dataset_args.chunk_size 512 \
        --eval_batch_size 512\
        --batch_size 64 \
        --model_args.use_char_context 1 \
        --model_checkpoint.save_model 1\
        --actually_stop_at 750000 \
        --n_train_iters 750000 \
        --debug 0 \
        $STUFF

###

STUFF="--config_file configs/instructions-wikipedia.yaml --mode run --group wiki --eval_every_batches 4096"
python main.py \
        --name base-wiki-llama2 \
        --model_args.input_tokenizer meta-llama/Llama-2-7b-hf \
        --dataset_args.chunk_size 512 \
        --eval_batch_size 512\
        --batch_size 64 \
        --model_args.use_char_context 0 \
        --model_checkpoint.save_model 1\
        --actually_stop_at 750000 \
        --n_train_iters 750000 \
        --debug 0 \
        $STUFF

STUFF="--config_file configs/instructions-wikipedia.yaml --mode run --group wiki --eval_every_batches 4096"
python main.py \
        --name ours-wiki-llama2 \
        --model_args.input_tokenizer meta-llama/Llama-2-7b-hf \
        --dataset_args.chunk_size 512 \
        --eval_batch_size 512\
        --batch_size 64 \
        --model_args.use_char_context 1 \
        --model_checkpoint.save_model 1\
        --actually_stop_at 750000 \
        --n_train_iters 750000 \
        --debug 0 \
        $STUFF
##############################################
##############################################

