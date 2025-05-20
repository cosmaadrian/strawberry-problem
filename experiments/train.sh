#!/bin/bash
export HF_TOKEN=`cat ./.hf_token`

cd ..
set -e

STUFF="--config_file configs/instructions-acumen.yaml --mode run --group main --eval_every_batches 4096"

for vocab_size in 128 256 512 1024 2048 4096 8192 16384 32768
do
    for K in 4 6 8
    do
        python main.py \
                --name ours-$vocab_size-$K \
                --dataset_args.chunk_size 512 \
                --model_args.input_tokenizer ./assets/tokenizers/acumen-tokenizer-$vocab_size-$K.json \
                --model_args.char_llm_args.dmodel 256\
                --dataset_args.dataset_name "" \
                --dataset_args.subset "" \
                --eval_batch_size 512\
                --batch_size 64 \
                --model_args.use_char_context 1 \
                --model_checkpoint.save_model 1\
                --actually_stop_at 750000 \
                --n_train_iters 750000 \
                --debug 0 \
                $STUFF

        python main.py \
                --name base-$vocab_size-$K \
                --dataset_args.chunk_size 512 \
                --model_args.input_tokenizer ./assets/tokenizers/acumen-tokenizer-$vocab_size-$K.json \
                --model_args.char_llm_args.dmodel 256\
                --dataset_args.dataset_name "" \
                --dataset_args.subset "" \
                --eval_batch_size 512\
                --batch_size 64 \
                --model_args.use_char_context 0 \
                --model_checkpoint.save_model 1\
                --actually_stop_at 750000 \
                --n_train_iters 750000 \
                --debug 0 \
                $STUFF
    done
done