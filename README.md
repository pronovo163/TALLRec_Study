Our weights for the instruction tuning model is uploading [here](https://drive.google.com/file/d/1teUwLm4BOqhngfCKKXE1tiMhJPf_FvRJ/view?usp=sharing)

**TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation** is available at https://arxiv.org/abs/2305.00447.

Train TALLRec base on LLaMA7B:
```
bash ./shell/instruct_7B.sh  gpu_id random_seed
```
If you want to run it under your environment, you need to make changes to the sh file:
- output_dir: Model save path，we will automatically add the seed and the sample to the end of the path for each experiments.
- base_model: LLaMA parameter weight path in Hugginface format
- train_data:  Training data path such as "./data/movie/train.json" for movie dataset.
- val_data: Validation data set path such as "./data/movie/valid.json" for movie dataset.
- instruction_model: The LoRA weights after the instruction tuning, for example lora weight from alpaca-lora.

After training, you need to evluate the test result on the best model evaluated by the validation set.
```
bash ./shell/evaluate.sh  gpu_id  output_dir
```
If you want to run it under your environment, you need to make changes to the sh file:
- base_model: LLaMA parameter weight path in Hugginface format
- test_data: Test data set path such as "./data/movie/test.json" for movie dataset.

Note that we will automatically detect all the different seed and sample files in the output_dir directory, and then integrate these results into the output_dir.json file.

Our project is developed based on the Alpaca_lora [repo](https://github.com/tloen/alpaca-lora), thanks for their contributions.

For "Environment setting sharing for CUDA 12.0", please see [here](https://github.com/SAI990323/TALLRec/issues/46).


训练：sudo bash ./shell/instruct_7B.sh 0 42
评估：sudo bash ./shell/evaluate.sh 0 ./model_42_64

可能的坑：

autodl上无法下载hf的模型，使用国内镜像也不可以，可能是autodl的ip被加黑名单了

解决方案：

1、手动去官网下载或者使用自己电脑下载后找到缓存的模型，压缩之后上传到autodl；
2、在autodl上解压缩模型，然后把模型移动到数据盘autodl-tmp（不移动的话，内存可能不足），文件重命名为decapoda-research-llama-7B-hf，其子文件名为models--decapoda-research-llama-7B-hf
3、autodl上使用的时候LlamaForCausalLM.from_pretrained 和 LlamaTokenizer.from_pretrained 要传cache_dir的参数，cache_dir='../autodl-tmp/decapoda-research-llama-7B-hf'，pretrained_model_name_or_path='decapoda-research-llama-7B-hf'

pytorch GPU不可用

解决方案：
vidia-smi 查看CUDA版本，然后安装与之匹配的pytorch版本

作者源代码evaluate.py中batch_size: int = 32 ，运行的时候会导致torch.cuda.OutOfMemoryError: CUDA out of memory，需要把batch_size调小，我调成15还不行，调成8就可以了
