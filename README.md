# RAGLens: Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders (ICLR 2026)

[![Homepage](https://img.shields.io/badge/Homepage-available-blue)](https://gzxiong.github.io/RAGLens/)
[![Arxiv](https://img.shields.io/badge/arXiv-2603.16737-b31b1b.svg)](https://arxiv.org/abs/2512.08892)
[![OpenReview](https://img.shields.io/badge/OpenReview-hgBZP67BkP-brightgreen)](https://openreview.net/forum?id=hgBZP67BkP)
[![Models](https://img.shields.io/badge/Models-available-yellow)](https://huggingface.co/collections/gzxiong/raglens)

## Overview

RAGLens is a lightweight hallucination detector for Retrieval‑Augmented Generation (RAG). It leverages sparse autoencoders (SAEs) to disentangle internal LLM activations and then applies a generalized additive model (GAM) over a small, information‑rich subset of features. This yields accurate faithfulness judgments and human‑readable rationales (global + token‑level), enabling practical post‑hoc mitigation.

<img src="figs/pipeline.png" alt="RAGLens pipeline: SAE-based hallucination detection, explanation, and mitigation for RAG" width="100%"/>

## Requirements

Install the dependencies listed in `requirements.txt`. Dependencies of the [sparsity](https://github.com/EleutherAI/sparsify) package are required for the training/inference of SAEs.

## Quickstart

The snippets below illustrate how the API works on a small, fast configuration. Please refer to the paper for the exact experimental settings.

### Setup

```python
import os
import sys
import torch
root_dir = './'
sys.path.append(os.path.join(root_dir, "src"))
from sparsify import Sae
from data_loading import RAGEvalDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import balanced_accuracy_score, f1_score

train_data = RAGEvalDataset("TofuEval", "dev", root_dir = './').items
train_labels = [1 if len(item['hall_info']) > 0 else 0 for item in train_data]
test_data = RAGEvalDataset("TofuEval", "test", root_dir = './').items
test_labels = [1 if len(item['hall_info']) > 0 else 0 for item in test_data]

llm_name = "meta-llama/Llama-3.2-1B"
sae_name = "EleutherAI/sae-Llama-3.2-1B-131k"
hookpoint = "layers.6.mlp"

tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=os.path.join(root_dir, "../huggingface/hub"))
model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, cache_dir=os.path.join(root_dir, "../huggingface/hub"), device_map="auto")
model.eval()
sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint, device="cuda")
sae.cfg.transcode = True if "transcoder" in sae_name else False
sae.eval()
```

### Detection

Fit the detector on the labelled training split, then flag unfaithful generations on the test split:

```python
from RAGLens import RAGLens

raglens = RAGLens(tokenizer=tokenizer, model=model, sae=sae, hookpoint=hookpoint)

raglens.fit(
    inputs = [item['input'] for item in train_data],
    outputs = [item['output'] for item in train_data],
    labels = train_labels,
)

preds = raglens.predict(
    inputs = [item['input'] for item in test_data],
    outputs = [item['output'] for item in test_data],
)
print(f"Balanced Accuracy: {balanced_accuracy_score(test_labels, preds)}") # 0.6865
print(f"Macro F1: {f1_score(test_labels, preds, average='macro')}") # 0.6876
```

### Interpretation

Because the predictor is a GAM over a small set of SAE features, each hallucination prediction decomposes into per-feature contributions, and each top contributor can be localized to the token where it fired. `raglens.explain(...)` returns these local explanations:

```python
explanations = raglens.explain(
    inputs = [item['input'] for item in test_data],
    outputs = [item['output'] for item in test_data],
)
```

### Mitigation

RAGLens can also revise outputs it flags as unfaithful. Two feedback modes are supported, matching the paper:

- `mode='instance'`: instance-level feedback that asks the model to revise its output based on the detector's decision.
- `mode='token'`: token-level feedback that additionally lists short spans around the tokens where the SAE features most responsible for the prediction fired.

```python
from mitigation import RAGLensMitigator, hf_generator

chat_llm_name = "meta-llama/Llama-3.2-1B-Instruct"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_llm_name, cache_dir=os.path.join(root_dir, "../huggingface/hub"))
chat_model = AutoModelForCausalLM.from_pretrained(chat_llm_name, torch_dtype=torch.bfloat16, cache_dir=os.path.join(root_dir, "../huggingface/hub"), device_map="auto")
chat_model.eval()

inputs = [item['input'] for item in test_data]
outputs = [item['output'] for item in test_data]

mitigator = RAGLensMitigator(raglens, hf_generator(chat_model, chat_tokenizer))
revised = mitigator.mitigate(inputs, outputs, mode='token')
```

For batched generation, use `vllm_generator`:

```python
from vllm import LLM
from mitigation import RAGLensMitigator, vllm_generator

llm = LLM(model=chat_llm_name, dtype="auto")
mitigator = RAGLensMitigator(raglens, vllm_generator(llm))
revised = mitigator.mitigate(inputs, outputs, mode='token')
```

## Citation
For the use of `RAGLens`, please consider citing
```bibtex
@inproceedings{xiong2026toward,
    title={Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders},
    author={Guangzhi Xiong and Zhenghao He and Bohan Liu and Sanchit Sinha and Aidong Zhang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=hgBZP67BkP}
}
```
