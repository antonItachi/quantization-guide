#!/bin/bash

# Define the scenarios and sequence lengths
SCENARIOS=("base" "amp" "flash")
SEQ_LENS=("256" "512" "1024")

# Loop through each scenario and sequence length
for scenario in "${SCENARIOS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        echo "Running scenario: $scenario with sequence length: $seq_len"
        python FA_AMP/test_scenarios.py --scenario $scenario --seq_len $seq_len
    done
done

echo "All tests completed!"