# All necessary import's
import torch
import argparse
from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import time
from torch.nn.utils.rnn import pad_sequence

import copy
import warnings
warnings.filterwarnings("ignore")


args = {
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
}


class AlpacaDataset(Dataset):
    def __init__(self, dataset, tokenizer, ignore_index=-100, seq_len=512):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.seq_len = seq_len
        self.prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        self.dataset = [
            item for item in dataset if len(self.tokenizer.encode(self._create_example(item))) >= self.seq_len
        ]

    def _create_example(self, ann):
        if ann.get("input", "") == "":
            return self.prompt_dict["prompt_no_input"].format_map(ann) + ann["output"]
        return self.prompt_dict["prompt_input"].format_map(ann) + ann["output"]

    def __getitem__(self, index):
        ann = self.dataset[index]
        example = self._create_example(ann)

        # Tokenization
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(example)] = self.ignore_index

        example_mask = example.ge(0)

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

    def __len__(self):
        return len(self.dataset)



# And to get the dataloader
def get_dataloader(tokenizer, seq_len, args):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned")['train']

    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(
        range(0, len(dataset) - len(dataset) % (args["batch_size"] * args["gradient_accumulation_steps"])))

    dataset = AlpacaDataset(dataset, tokenizer, seq_len=seq_len)

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        pad_token_id = tokenizer.eos_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)[:, :seq_len]
        if with_attention_mask:
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:,
                              :args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[:, :seq_len]

        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn)

    return dataloader


def get_memory_stats(device: torch.device, reset_stats: bool = True) -> dict:
    if device.type != "cuda":
        return {}
    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_mem_alloc,
        "peak_memory_reserved": peak_mem_reserved,
    }


def train_model(
    model,
    dataloader,
    optimizer,
    max_steps=400,
    use_amp=False,
    device="cuda",
):
    stats = []
    scaler = GradScaler() if use_amp else None
    model.to(device)
    model.train()
    step_counter = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.autocast(enabled=use_amp, device_type="cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tokens_per_second = input_ids.numel() / (time.time() - start_time)
        memory_stats = get_memory_stats(torch.device(device))
        stats.append({
            "step": step_counter + 1,
            "loss": 0 if torch.isnan(loss) else loss.item(),
            "tokens_per_second": tokens_per_second,
            **memory_stats,
        })
        step_counter += 1

        if step_counter >= max_steps:
            break

    return stats


import os
import json

def log_to_file(filepath, seq_len, scenario_name, stats):
    """
    Log results hierarchically by sequence length and scenario type.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Load existing data or initialize empty structure
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        data = {}
    else:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("JSONDecodeError: Creating a new file structure.")
                data = {}

    # Initialize nested dictionaries if necessary
    if seq_len not in data:
        data[seq_len] = {}
    if scenario_name not in data[seq_len]:
        data[seq_len][scenario_name] = []

    # Append stats to the respective section
    data[seq_len][scenario_name].extend(stats)

    # Write updated data back to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)



def main():
    parser = argparse.ArgumentParser(description="Running Test.")
    parser.add_argument(
        "--scenario",
        choices=["base", "amp", "flash"],
        required=True,
        help="Choose scenario: base (Base HF Model), amp (with AMP), flash (Flash Attention + AMP)",
    )
    parser.add_argument(
        "--seq_len",
        choices=["256", "512", "896", "1024"],
        required=True,
        help="Specify the sequence length: 256, 512, or 1024 tokens.",
    )
    args_test = parser.parse_args()

    seq_len = args_test.seq_len
    scenario = args_test.scenario
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    dataloader = get_dataloader(tokenizer, seq_len=int(seq_len), args=args)

    # Select model and run tests
    if scenario == "base":
        model = GPT2LMHeadModel.from_pretrained(model_id)
        optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        stats = train_model(model, dataloader, optimizer, use_amp=False)
    elif scenario == "amp":
        model = GPT2LMHeadModel.from_pretrained(model_id)
        optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        stats = train_model(model, dataloader, optimizer, use_amp=True)
    elif scenario == "flash":
        model = GPT2LMHeadModel.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2"
        )
        optimizer = AdamW(model.parameters(), lr=1e-6)
        stats = train_model(model, dataloader, optimizer, use_amp=True)

    # Log results
    log_to_file(
        filepath=f"FA_AMP/logs/flash_attention_amp_gpt2.json",
        seq_len=seq_len,
        scenario_name=scenario,
        stats=stats
    )

if __name__ == "__main__":
    import os

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()

