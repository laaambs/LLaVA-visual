#!/bin/bash
python test.py >> response_7b_val_92_new.out 2>&1
# python scripts/merge_lora_weights.py --model-path "/data/coding/models/llava-v1.5-7b-vcr-lora3" \
#        --model-base "/data/coding/models/llava-v1.5-7b" \
#        --save-model-path "/data/coding/models/llava-v1.5-7b-vcr-lora3-merged"
