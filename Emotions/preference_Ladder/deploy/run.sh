#!/bin/bash

python -u test.py \
    --device            cuda \
    --data_type         clean \
    --model_type        wav2vec2 \
    --batch_size        1 \
    --num_layers        2 \
    --hidden_dim        1024 \
    --model_path        model || exit 1;
