# Part 1: AMP and FlashAttention

This repository demonstrates how to optimize model training using **Automatic Mixed Precision (AMP)** and **FlashAttention** for modern transformer models like GPT-2. The code allows you to test performance improvements in terms of speed and memory efficiency.

All insights and explanations are available in my **Medium article**:  
[Optimizing Transformer Training with AMP and FlashAttention](https://medium.com/@krasniuk-ai/optimizing-transformer-training-with-amp-and-flashattention-gpt2-alpaca-1badd531d185)

---

## Folder Overview

- **FA_AMP/**: Contains the main implementation and test scripts for AMP and FlashAttention experiments.
   - `README.md`: This guide to help you understand the structure and usage.
   - `requirements.txt`: Dependencies required to run the project.
   - `test_scenarios.py`: Script for running performance tests on GPT-2 with different configurations.
   - `run_all_tests.sh`: A shell script to run all test cases for various configurations and input lengths.
   - `logs/`: Contains the generated JSON logs with test results.
   - `log-analysis.ipynb`: A Jupyter Notebook for analyzing and visualizing results from the logs.

---

## Installation

Before running the code, ensure you have a Python environment with the necessary libraries.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/antonItachi/quantization-guide.git
   cd quantization-guide/FA_AMP

## How to Run the Tests

The primary script is `test_scenarios.py`, which allows you to test the GPT-2 model under various configurations:

1. **Run the script for a specific scenario**:

   ```bash
   python test_scenarios.py --scenario {base|amp|flash} --seq_len {256|512|896|1024}

- **`--scenario`**: Specifies the configuration to test:
   - `base`: Standard GPT-2 model without optimizations.
   - `amp`: GPT-2 model using Automatic Mixed Precision.
   - `flash`: GPT-2 model with FlashAttention + AMP.
- **`--seq_len`**: Specifies the input sequence length: 256, 512, 896, or 1024.

**Example**:
   ```bash
   python test_scenarios.py --scenario amp --seq_len 512 
   ```

2. **Run all tests automatically**: 
Use the provided shell script to execute all combinations of scenarios and sequence lengths:

```bash
   bash run_all_tests.sh 
```

This script sequentially runs:
   - Base model for all sequence lengths.
   - AMP model for all sequence lengths.
   - FlashAttention with AMP for all sequence lengths.

## Analyzing Results

`Logs`: After running the tests, results are stored in logs/flash_attention_amp_gpt2.json.

`Visualization`: Use log-analysis.ipynb to analyze and visualize the performance results:
   - Average tokens per second.
   - Active and reserved memory usage.
   - Steps completed for each configuration.

