#!/bin/bash
export HF_TOKEN=`cat ./.hf_token`

cd ..
set -e

cd scripts/
bash generate.sh