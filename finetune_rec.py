# coding=utf-8
import os
# 设置环境变量LD_LIBRARY_PATH: 当系统默认的库路径不包含需要的库时,在程序运行时搜索共享库的目录
# os.environ['LD_LIBRARY_PATH'] = '/data/baokq/miniconda3/envs/alpaca_lora/lib/'
import sys
import fire
import torch
import transformers
from typing import List
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import EarlyStoppingCallback
from util.utils import preprocess_logits_for_metrics, compute_metrics, generate_and_tokenize_prompt

# peft 库中含有包括LoRA在内的多种高效微调方法
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


def train(
        # model/data params
        base_model: str = "baffo32/decapoda-research-llama-7B-hf",  # the only required argument
        train_data_path: str = "./data/movie/train.json",
        val_data_path: str = "./data/movie/valid.json",
        output_dir: str = "./lora-alpaca",
        sample: int = -1,
        seed: int = 0,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # 计算梯度累积步数
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    # PyTorch中分布式数据并行（Distributed Data Parallel，DDP）的设置
    device_map = "auto"  # device_map是一个字典，用于指定每个进程应该使用的GPU设备;当设置为"auto"时，PyTorch会自动为每个进程分配一个可用的GPU
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # WORLD_SIZE表示参与分布式训练的总进程数, 默认为1
    ddp = world_size != 1  # 是否启用分布式数据并行
    # 若启用分布式训练，创建device_map字典指定当前进程应该使用的GPU设备，重新指定在更新模型权重之前应该累积多少个批次（batch）的梯度
    if ddp:
        # LOCAL_RANK标识同一节点上的不同进程，空字符串""作为键表示默认设备，而整数值表示具体的GPU编号
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # {"": 0}、{"": 1}
        # 使用多进程时，每个进程都会处理一部分数据，更新累积步数避免过多计算
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # 设置实验追踪和模型管理Weights & Biases（WandB）环境变量
    # use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # 加载预训练的Llama，因果语言建模（Causal Language Modeling）基于给定的文本上下文生成下一个最可能的词
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 加速 huggingface模型下载

    model = LlamaForCausalLM.from_pretrained(
        base_model,  # 预训练模型的名称或包含模型权重的本地路径。如果是本地路径，函数会从该路径加载模型权重；如果是模型名称，函数会自动从 Hugging Face 的模型仓库下载对应的权重
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (0)  # 使用ID为0的token进行padding
    tokenizer.padding_side = "left"  # 从左侧填充

    # LoRA相关配置
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # 如果提供了checkpoint路径，则从checkpoint恢复模型
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # 构造检查点的完整路径
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint,
                                           "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (False)  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    # 没有使用DDP且有多于一个的CUDA设备,开启模型并行,通常是单个模型太大而无法完全放入单个GPU内存中，模型的不同部分被分配到不同的GPU上
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # 禁用Weights & Biases 可能是为了减少计算资源的开销，或不需要实验跟踪
    os.environ["WANDB_DISABLED"] = "true"

    # 加载训练和验证数据集
    if train_data_path.endswith(".json"):
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)

    if val_data_path.endswith(".json"):
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    # 采样及打散训练样本，train_data和val_data是字典，对其中"train"键对应的数据集进行操作
    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data[
        "train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    # 数据预处理和转换
    train_data = (
        train_data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len, train_on_inputs)))
    val_data = (
        val_data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len, train_on_inputs)))

    if sample > -1:
        if sample <= 128:
            eval_step = 10
        else:
            eval_step = sample / 128 * 5

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,  # 每个设备（如GPU或CPU）上的批量大小
            gradient_accumulation_steps=gradient_accumulation_steps,  # 梯度累积的步数
            warmup_steps=20,  # 在学习率预热阶段，学习率将线性增加到其初始值的步数
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,  # 每多少步记录一次日志
            optim="adamw_torch",
            evaluation_strategy="steps",  # 评估策略，'no'（不评估）、'steps'（每eval_steps步评估一次）、'epoch'（每个epoch结束时评估一次）
            save_strategy="steps",  # 保存策略，同上
            eval_steps=eval_step,  # 每多少步进行一次评估
            save_steps=eval_step,  # 每多少步保存一次模型检查点
            output_dir=output_dir,  # 训练过程中生成的文件和模型检查点将被保存在这个目录下
            save_total_limit=1,  # 最大保存的检查点数量
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            # eval_accumulation_steps=10,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,  # 与模型相对应的，对输入和目标文本进行编码的分词器实例
            pad_to_multiple_of=8,
            return_tensors="pt",  # 返回的数据类型，可以是 'pt' (PyTorch tensors) 或 'tf' (TensorFlow tensors)
            padding=True
        ),
        compute_metrics=compute_metrics,  # 字典类型的评估指标
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 在计算指标之前对模型输出的原始 logits 进行预处理
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  # 连续 10 次评估都没有提升，训练将被提前终止
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict  # 模型在某个时间点的完整状态，包含模型所有权重和偏置的 Python 字典对象，通常用于模型的保存和加载
    # 使用get_peft_model_state_dict函数处理状态字典
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model,
                                                                                                          type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # 是否从checkpoint恢复模型

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
