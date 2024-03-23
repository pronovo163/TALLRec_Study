# coding=utf-8
import os
import sys
import fire
import torch
import json
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from util.utils import generate_prompt_for_evaluate, get_batch

torch.set_num_threads(1)  # 将PyTorch 所使用的线程数设置为 1
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 控制 OpenBLAS（用于线性代数运算优化的BLAS库）的线程数设置为1
os.environ['OMP_NUM_THREADS'] = '1'  # 控制 OpenMP（支持多平台共享内存并行编程的API）的线程数设置为1

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # MPS 是 Apple 设备上用于 GPU 计算的 API
    device = "mps"  # 在 Apple 的 GPU 上运行
else:
    device = "cpu"


def main(
        load_8bit: bool = False,
        base_model: str = "baffo32/decapoda-research-llama-7B-hf",
        lora_weights: str = "./model_0_64",
        test_data_path: str = "./data/movie/test.json",
        result_json_data: str = "temp.json",
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"load_8bit: {load_8bit}\n"
        f"base_model: {base_model}\n"
        f"lora_weights: {lora_weights}\n"
        f"test_data_path: {test_data_path}\n"
        f"result_json_data: {result_json_data}\n")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])

    if 'book' in model_type:
        train_sce = 'book'
    else:
        train_sce = 'movie'

    if 'book' in test_data_path:
        test_sce = 'book'
    else:
        test_sce = 'movie'

    temp_list = model_type.split('_')
    seed = temp_list[-2]
    sample = temp_list[-1]

    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if train_sce not in data:
        data[train_sce] = {}
    if test_sce not in data[train_sce]:
        data[train_sce][test_sce] = {}
    if model_name not in data[train_sce][test_sce]:
        data[train_sce][test_sce][model_name] = {}
    if seed not in data[train_sce][test_sce][model_name]:
        data[train_sce][test_sce][model_name][seed] = {}
    if sample in data[train_sce][test_sce][model_name][seed]:
        exit(0)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else {"": device},
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={'': 0} if device == "cuda" else {"": device},
        torch_dtype=torch.float16
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # 模型和分词器使用token_id为0的进行填充
    model.config.bos_token_id = 1  # 序列开始（Begin Of Sequence，BOS）标记用于指示序列的开始
    model.config.eos_token_id = 2  # 序列结束（End Of Sequence，EOS）标记用于指示序列的结束

    if not load_8bit:
        model.half()  # 半精度运算，将模型的权重和激活值转换为半精度浮点数（16位浮点数，float16），减少模型的内存占用和加速计算

    model.eval()  # 将模型设置为评估模式，注意某些层（如 dropout 和 batch normalization）的运行会与训练模式不同

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instructions,
            inputs=None,
            temperature=0,
            top_p=1.0,
            top_k=40,
            num_beams=1,
            max_new_tokens=128,
            **kwargs,
    ):
        prompt = [generate_prompt_for_evaluate(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:, [8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]

        return output, logits.tolist()

    outputs = []
    logits = []
    pred = []

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]

        for i, batch in tqdm(enumerate(zip(get_batch(instructions), get_batch(inputs)))):
            instructions, inputs = batch
            output, logit = evaluate(instructions, inputs)
            outputs = outputs + output
            logits = logits + logit
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            pred.append(logits[i][0])

    data[train_sce][test_sce][model_name][seed][sample] = roc_auc_score(gold, pred)
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()


if __name__ == "__main__":
    fire.Fire(main)
