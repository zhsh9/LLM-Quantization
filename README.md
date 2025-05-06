
# Qwen3 Model Quantization Toolkit

## Project Overview

This project provides various quantization method implementations for the Qwen3 large language model. Quantization can significantly reduce model size and inference time while maintaining model performance as much as possible. We support five advanced quantization techniques, each with its unique advantages and application scenarios.

## Supported Quantization Methods

| Method | Description | Documentation |
|--------|-------------|---------------|
| [AWQ](llm-awq) | Activation-aware Weight Quantization, an efficient quantization method for LLM compression and acceleration | [AWQ Documentation](llm-awq/readme.md) |
| [GPTQ](GPTQ-for-Qwen) | Gradient-based Post-training Quantization method  | [GPTQ Documentation](GPTQ-for-Qwen/README.md) |
| [RTN](RTN)| Recursive Tensor Network quantization method | [RTN Documentation](RTN/README.md) |
| [SmoothQuant](SmoothQuant-for-Qwen3) | Smooth quantization technique that reduces quantization error propagation | [SmoothQuant Documentation](SmoothQuant-for-Qwen3/README.md) |
| Bi-LLM | Dual-precision quantization method that maintains high precision in critical layers | [Bi-LLM Documentation](./bi-llm.md) |

## Attention!!!
### File Upload Limitation Notice

Due to GitHub's file size restrictions, the following five files cannot be directly uploaded to our code repository:

- `eval_my/mmlu/data/auxiliary_train/race.csv`
- `eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/test`
- `eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train`
- `eval_my/ppl_datasets/allenai/c4/allenai--c4/train`
- `eval_my/ppl_datasets/allenai/c4/allenai--c4/validation`

You can download these files from Hugging Face. We will also update them to our project's Hugging Face repository in the future.
## Contributions

Contributions to this project are welcome! If you have suggestions for improvements or find issues, please submit an issue or pull request.



