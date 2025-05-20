#!/bin/bash
export HF_TOKEN=`cat ./.hf_token`

cd ..
set -e

# Dmodel variation
STUFF="--config_file configs/instructions-acumen.yaml --mode run --group dmodel-ablation --eval_every_batches 4096"
for dmodel in 64 128
do 
    python main.py \
            --name ours-tok-dmodel-$dmodel-8192-4 \
            --dataset_args.chunk_size 512 \
            --model_args.input_tokenizer ./assets/tokenizers/acumen-tokenizer-8192-4.json \
            --model_args.char_llm_args.dmodel $dmodel\
            --model_args.char_llm_args.context_position 3\
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
done

# Position variation
STUFF="--config_file configs/instructions-acumen.yaml --mode run --group position-ablation --eval_every_batches 4096"
for position in 0 3 7
do 
    python main.py \
            --name ours-tok-position-$position-8192-4 \
            --dataset_args.chunk_size 512 \
            --model_args.input_tokenizer ./assets/tokenizers/acumen-tokenizer-8192-4.json \
            --model_args.char_llm_args.dmodel 256\
            --model_args.char_llm_args.context_position $position\
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
done
