import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
)
from transformers import AutoTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant import fake_quant

import tqdm

import random

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, LlamaTokenizer
import os

DEV = torch.device('cuda:0')

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    alldata = load_from_disk('/home/beihang/lin/pretrained_models/c4/allenai--c4')
    traindata = alldata['train']
    valdata = alldata['validation']

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, help='qwen model to load')
parser.add_argument('--weight',type=int)
parser.add_argument('--activation',type=int)
parser.add_argument('--model_name',type=str)
parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')

args = parser.parse_args()

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(args.model)
model_fp16 = Qwen3ForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map="auto"
)



fake_quant.activate_bit = args.activation
fake_quant.weight_bit = args.weight

model_quantized = fake_quant.quantize_llama_like(model_fp16)
print(model_quantized)

if args.eval:
    from eval_my import evaluate_
    evaluate_.device = DEV
    evaluate_.eval_ours(model_quantized, tokenizer,model_name=f"{args.model_name}_w{fake_quant.weight_bit}a{fake_quant.activate_bit}")
