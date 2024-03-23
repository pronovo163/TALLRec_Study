# coding=utf-8
import torch
from sklearn.metrics import roc_auc_score


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    return {'auc': auc}

def get_batch(list, batch_size=32):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

def tokenize(tokenizer, prompt, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len,train_on_inputs):
    full_prompt = generate_prompt_for_finetune(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len)
    if not train_on_inputs:
        user_prompt = generate_prompt_for_finetune({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(tokenizer, user_prompt, cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                     user_prompt_len:]  # could be sped up, probably
    return tokenized_full_prompt


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
    labels_index[:, 1] = labels_index[:, 1] - 1
    logits = logits.softmax(dim=-1)
    logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:, [3782, 8241]], dim=-1)
    return logits[:, 1][2::3], gold[2::3]


def generate_prompt_for_finetune(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

def generate_prompt_for_evaluate(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""
