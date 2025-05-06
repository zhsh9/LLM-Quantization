# SmoothQuant: **Smooth Quantization** for LLM quantization (Qwen3)

## Installation Setup

1. Clone the our repository and open the `smoothquant-main` directory in `SmoothQuant-for-Qwen`.Use the following command to install our modified SmoothQuant library.

```bash
cd SmoothQuant-for-Qwen/smoothquant-main
conda create -n smoothquant python=3.10 -y
conda activate smoothquant
python setup.py install
```

2. Install required packages

```bash
pip install --upgrade pip  # enable PEP 660 support
```

## File Replacement

Before starting quantization, you need to replace some files to support the Qwen3 model:

- **Add eval_my directory**: Place the `eval_my` directory under the `SmoothQuant-for-Qwen` directory.

## Quantization and Evaluation

### Per-channel Quantization

#### 1. Perform SmoothQuant search

```bash
CUDA_VISIBLE_DEVICES=0 python SmoothQuant-for-Qwen/smoothquant_qwen.py \
path_of_model --weight 8 --activation 8 \
--model_name Qwen3-4B 
```

#### 2. Perform SmoothQuant search and Evaluate the quantized model

```bash
CUDA_VISIBLE_DEVICES=0 python SmoothQuant-for-Qwen/smoothquant_qwen.py \
path_of_model --weight 8 --activation 8 \
--model_name Qwen3-4B --eval
```

## Notes

- Available quantization bit-widths : (w_bit):2,4,8; (a_bit):4,8.
- The default is per-channel quantization.
- The `model_name`  parameter is chosen as a suffix for storing the result tables of MMLU. It is not very important.
- Make sure you have sufficient GPU memory to run a 32B-sized model