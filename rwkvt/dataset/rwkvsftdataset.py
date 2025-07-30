
import copy
from typing import Optional, Dict, Sequence, List, Literal
import logging
import torch.nn.functional as F

import torch
import torch.distributed
import transformers
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass, field
# from transformers import AutoTokenizer, Rwkv6Config, Rwkv6Model, Rwkv6Tokenizer

#tokenizer = Rwkv6Tokenizer.from_pretrained("RWKV/v6-Finch-1B6-HF")

tokenizer_path = 'RWKV/rwkv-5-world-3b'
IGNORE_INDEX = -100
EOT_TOKEN = "\x17"

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def process_conversation_text(conversations, tokenizer):
    input_ids = []
    labels = []
    conversation_text = ""

    for conv in conversations:
        role = conv.get('from', '').lower()
        content = conv.get('value', '')
        
        if role == 'human':
            conversation_text += f"User: {content}"
            token = tokenizer.decode(conversation_text)
            token_length = len(token)
            token_mask = torch.full((token_length,), IGNORE_INDEX)

        elif role in ['assistant', 'gpt']:
            conversation_text += f"Assistant: {content}"
            token = tokenizer.decode(conversation_text)
            token_mask = token
        input_ids.append(token)
        labels.append(token_mask)
    return {
        "input_ids": input_ids,
        "labels": labels
    }

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def sft_dataset(script_args):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=script_args.ctx_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    # logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    # logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    raw_train_datasets = load_dataset(script_args.data_file, split=script_args.sft_split)

    # if script_args.local_rank > 0: 
    #     torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.sft_field[0], "response": script_args.sft_field[1]}
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = data_collator(train_dataset)

    return (data_module["input_ids"], data_module["labels"], data_module["attention_mask"])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
# class Data():
#     model_name_or_path: str = "RWKV/rwkv-5-world-3b"
#     data_file :str = "/home/rwkv/JL/data/MetaMathQA"
#     sft_split: str = "train[:5000]"#field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
#     sft_field: List[str] = ["query", "response"]#field(default=["query", "response"], metadata={"help": "Fields of dataset input and output."})
#     ctx_len: int = 100


# if __name__ == "__main__":
#     args = Data()
#     sft_dataset(args)

