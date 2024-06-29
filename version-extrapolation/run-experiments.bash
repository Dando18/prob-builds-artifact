#!/bin/bash

# settings
SEED="42"
DROP="0.2" # what percent of samples in test set (i.e. need to be extrapolated)
CV="5"
NORM_METHOD="maxabs"
JOBS="4"
OUTPUT_ROOT="results"

# args to enumerate
NORMALIZE=("--normalize per-pair" "--normalize per-parent" "--normalize per-child" \
           "--normalize per-parent-version" "--normalize per-child-version" "--normalize all" "")

# setup environment
source .env/bin/activate

# run experiments
for norm in "${NORMALIZE[@]}"; do
    norm_pretty=$(echo $norm | sed 's/--normalize //g')
    if [ "$norm_pretty" == "" ]; then
        norm_pretty="none"
    fi

    dirname=$(printf "norm_%s" "$norm_pretty")
    OUTPUT_DIR="$OUTPUT_ROOT/$dirname"
    
    mkdir -p "$OUTPUT_DIR"

    python evaluate-methods.py \
        --seed $SEED \
        --drop $DROP \
        --cv $CV \
        --normalization-method $NORM_METHOD \
        -j $JOBS \
        $norm \
        --output-root $OUTPUT_DIR
done
