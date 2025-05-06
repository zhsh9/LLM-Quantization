# RTN: **Round to Nearest **Quantization (Qwen3)

> The quantization and evaluation of RTN are encapsulated in the GPTQ method. By following the same installation steps as GPTQ, you can run RTN quantization and perform evaluations.

## Installation Setup

1. You can clone the GPTQ-for-LLaMA repository and Replace the `qwen.py` file in the `GPTQ-for-Qwen` directory.

```bash
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
```

​	For more details about the GPTQ-for-LLaMA repository, please refer to [this link](https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/README.md).

​	OR You can simply clone our repository, the directory `GPTQ-for-Qwen` can be directly 	run and used.

2. Install required packages

```bash
conda create -n gptq python=3.10 -y
conda activate gptq
pip install --upgrade pip  # enable PEP 660 support
```

## File Replacement

Before starting quantization, you need to replace some files to support the Qwen3 model:

- **Add eval_my directory**: Place the `eval_my` directory under the `GPTQ-for-Qwen` directory.

## Quantization and Evaluation

### Group-wise Quantization

#### 1. Perform RTN search and save results

```bash
CUDA_VISIBLE_DEVICES=0 python path_of_qwen.py your_model_path c4(Validation dataset) \
--wbits 4 --true-sequential --act-order \
--groupsize 128  --model_name gptq_4B_w4_128 \
--save gptq_4B_w4_128.pth --nearest
```

#### 2. Evaluate the quantized model

```bash
CUDA_VISIBLE_DEVICES=0 python path_of_qwen.py your_model_path c4(Validation dataset) \
--wbits 4 --true-sequential --act-order \
--groupsize 128  --load gptq_4B_w4_128.pth \
--eval --nearest
```

### Per-channel Quantization

#### 1. Perform RTN search and save results 

```bash
CUDA_VISIBLE_DEVICES=0 python path_of_qwen.py your_model_path c4(Validation dataset) \
--wbits 4 --true-sequential --act-order \
--groupsize -1  --model_name gptq_4B_w4_128 \
--save gptq_4B_w4_128.pth --nearest
```

#### 2. Evaluate the quantized model

```bash
CUDA_VISIBLE_DEVICES=0 python path_of_qwen.py your_model_path c4(Validation dataset) \
--wbits 4 --true-sequential --act-order \
--groupsize -1  --load gptq_4B_w4_128.pth \
--eval --nearest
```

## Notes

- Available quantization bit-widths (w_bit): 2, 4, 8.
- Use the `--groupsize` parameter (e.g., 128) for group-wise quantization
- Set the `groupsize` parameter to -1 for per-channel quantization
- The `model_name`  parameter is chosen as a suffix for storing the result tables of MMLU. It is not very important.
- Make sure you have sufficient GPU memory to run a 32B-sized model