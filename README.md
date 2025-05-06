
# Qwen3 Model Quantization Toolkit

## Project Overview

This project provides various quantization method implementations for the Qwen3 large language model. Quantization can significantly reduce model size and inference time while maintaining model performance as much as possible. We support five advanced quantization techniques, each with its unique advantages and application scenarios.

## Supported Quantization Methods

| Method | Description | Documentation |
|--------|-------------|---------------|
| AWQ | Activation-aware Weight Quantization, an efficient quantization method for LLM compression and acceleration | [AWQ Documentation](llm-awq/readme.md) |
| GPTQ | Gradient-based Post-training Quantization method,  | [GPTQ Documentation](GPTQ/GPTQ-for-Qwen(and RTN)/READ.md) |
| RTN | Recursive Tensor Network quantization method | [RTN Documentation](GPTQ/GPTQ-for-Qwen(and RTN)/README.md) |
| SmoothQuant | Smooth quantization technique that reduces quantization error propagation | [SmoothQuant Documentation](SmoothQuant-for-Qwen3/README.md) |
| Bi-LLM | Dual-precision quantization method that maintains high precision in critical layers | [Bi-LLM Documentation](./bi-llm.md) |

## Contributions

Contributions to this project are welcome! If you have suggestions for improvements or find issues, please submit an issue or pull request.

## License

Please refer to the LICENSE file in the project root directory for details.
