# BiLLM: Pushing the Limit of Post-Training Quantization for LLMs (Qwen3)

## Installation Setup

1. Clone the BiLLM repository and navigate to the BiLLM folder
```bash
git https://github.com/Aaronhuang-778/BiLLM.git
cd BiLLM
```

2. Dependence
```bash
torch==2.0.1+cu117
transformers==4.51.3
datasets==2.14.6
huggingface-hub==0.16.4
```

## File Replacement

Before starting quantization, you need to replace some files to support the Qwen3 model:

1. **Add eval_my directory**: Place the eval_my directory at the same level as the BiLLM folder.

2. **Replace bigptq.py file**: Replace the file at BiLLM/bigptq.py with our modified version. 

3. **Replace run.py files**: Replace the files at BiLLM/run.py with our versions.

## Quantization and Evaluation

### Quantized and evaluate model
```bash
CUDA_VISIBLE_DEVICES=0, python3 run.py /Path/to/Qwen3/Qwen3-32B c4 braq --blocksize 128 --save --salient_metric hessian --device "cuda:0" | tee billm_qwen3_14B.log
```