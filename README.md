# Learned Transfer Membership Inference Attack

A classifier that detects whether a given text was part of a language model's fine-tuning data. It compares the output distributions of a fine-tuned model against its pretrained base, extracting per-token features that a small transformer classifier uses to predict membership. Trained on 10 transformer models × 3 text domains, it generalizes zero-shot to unseen model/dataset combinations, including non-transformer architectures (Mamba, RWKV, RecurrentGemma).

**Pre-trained model available on HuggingFace:** [JetBrains-Research/learned-transfer-attack](https://huggingface.co/JetBrains-Research/learned-transfer-attack)

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended for feature extraction and training)

## Installation

```bash
git clone https://github.com/JetBrains-Research/learned-mia.git
cd learned-mia
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch transformers datasets peft scikit-learn numpy pyyaml tqdm huggingface_hub
```

## Directory Structure

Before running, create the required directories:

```bash
mkdir -p data/features
mkdir -p outputs
```

## Quick Start

### 1. Extract Features

Extract features for a single model/dataset combination:

```bash
python extract_features.py --config ../configs/extract_training_data.yaml
```

Or run batch extraction for all combinations:

```bash
python run_all_extractions.py --config ../configs/extract_training_data.yaml
```

Options:
- `--skip-existing`: Skip combinations already in the manifest
- `--dry-run`: Print what would be done without executing
- `--only combo_id [combo_id ...]`: Only run specific combinations
- `--no-clear-cache`: Disable automatic HuggingFace cache clearing between models

### 2. Train MIA Classifier

Train the attack model on extracted features:

```bash
python train_combined.py --config ../configs/train_attacker.yaml
```

Options:
- `--manifest PATH`: Override manifest path
- `--save-path PATH`: Override checkpoint save path
- `--epochs N`: Override training epochs
- `--batch-size N`: Override batch size
- `--lr FLOAT`: Override learning rate

### 3. Evaluate

Evaluate the trained classifier:

```bash
python evaluate.py --config ../configs/evaluate_attacker.yaml
```

Options:
- `--checkpoint PATH`: Override checkpoint path
- `--combos combo_id [combo_id ...]`: Evaluate specific combinations
- `--split {train,val,test}`: Data split to evaluate on

## Configuration Files

Configuration files are in YAML format. Key sections:

### Feature Extraction (`extract_training_data.yaml`)

```yaml
output_root: data/features
manifest_path: data/manifest.yaml

defaults:
  seed: 42
  n_members: 10000
  n_nonmembers: 10000
  n_ft_val: 500
  sequence_length: 128
  top_k: 20
  ft_epochs: 3
  ft_batch_size: 16
  ft_lr: 5e-5
  inference_batch_size: 16
  val_ratio: 0.05
  test_ratio: 0.05

combinations:
  - model: gpt2
    dataset: ag_news
    lora:
      use_lora: false
```

### Training (`train_attacker.yaml`)

```yaml
manifest_path: data/manifest.yaml
save_path: outputs/mia_combined.pt

train_combinations:
  - gpt2_ag_news
  - gpt2_wikipedia

mia_model:
  architecture: transformer
  d_model: 192
  nhead: 8
  num_layers: 2
  dim_ff: 576
  dropout: 0.05

train:
  epochs: 80
  batch_size: 16384
  lr: 9.88e-4
  balance_strategy: uniform
  label_smooth: 0.02
  lr_scheduler: cosine
  warmup_epochs: 10
  weight_decay: 0.028
  grad_clip: 1.0
```

### Evaluation (`evaluate_attacker.yaml`)

```yaml
eval:
  checkpoint: outputs/mia_combined.pt
  manifest_path: data/manifest.yaml
  combinations:
    - gpt2_ag_news
  split: test
  batch_size: 8192
  per_combo_breakdown: true
  results_path: results.csv
```

## Supported Models

The framework supports various model architectures:

- **GPT-2 family**: distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
- **Pythia**: EleutherAI/pythia-* (various sizes)
- **LLaMA-style**: meta-llama/Llama-2-*, openlm-research/open_llama_*
- **Falcon**: tiiuae/falcon-*
- **Gemma**: google/gemma-*
- **Mamba**: state-spaces/mamba-*
- **RWKV**: RWKV/rwkv-*
- **And more**: MPT, BLOOM, OLMo, StableLM, Phi, Mistral, OPT

## Supported Datasets

- **News**: ag_news, news-category-dataset
- **Wikipedia**: wikipedia (streaming)
- **Summarization**: xsum, cnndm
- **Academic**: arxiv (streaming)
- **Code**: swallow_code
- **General**: wikitext-103

## MIA Model Architectures

Available architectures for the attack classifier:

- `transformer`: Transformer encoder with attention pooling (default)
- `pooled_transformer`: Pools features before transformer processing
- `mlp`: Multi-layer perceptron (flattens sequence)
- `mean_mlp`: MLP on sequence mean
- `lr`: Logistic regression baseline

## Output Metrics

The evaluation produces:
- **Accuracy**: Classification accuracy
- **AUC**: Area under ROC curve
- **TPR@1%FPR**: True positive rate at 1% false positive rate
- **TPR@0.1%FPR**: True positive rate at 0.1% false positive rate

## Feature Representation

Per-token features (dimension: 7×k + 14 where k=top_k) include:
- Ground truth logits from target and reference models
- Top-k and bottom-k logit comparisons
- Cross-model ranking information
- Per-token and sequence-level loss statistics

## Quick Inference

Use the pre-trained model from HuggingFace:

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from ltmia import extract_per_token_features_both, create_mia_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load your base and fine-tuned models
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_ref = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
model_tgt = AutoModelForCausalLM.from_pretrained("./my-finetuned-gpt2").to(device).eval()

# 2. Extract features
texts = ["Text you want to check...", "Another text..."]

feats, masks, _ = extract_per_token_features_both(
    model_tgt, model_ref, tokenizer, texts,
    device=device, batch_size=8, sequence_length=128, k=20,
)

# 3. Load the MIA classifier
ckpt_path = hf_hub_download(
    repo_id="JetBrains-Research/learned-transfer-attack",
    filename="mia_combined_400k.pt",
)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
mia = create_mia_model(
    architecture=ckpt["architecture"],
    d_in=ckpt["d_in"],
    seq_len=ckpt.get("seq_len", 128),
    **ckpt["mia_hparams"],
)
mia.load_state_dict(ckpt["state_dict"])
mia.to(device).eval()

# 4. Predict membership
with torch.no_grad():
    logits = mia(
        torch.from_numpy(feats).to(device),
        torch.from_numpy(masks).to(device),
    )
    probs = torch.sigmoid(logits)

for text, p in zip(texts, probs):
    prob = p.item()
    label = "MEMBER" if prob > 0.5 else "NON-MEMBER"
    print(f"[{prob:.4f}] {label}  ←  {text[:80]}")
```

## Evaluation (Out-of-Distribution)

Performance on models and datasets **never seen** during classifier training:

| Architecture | Model | Dataset | AUC |
|---|---|---|---|
| Transformer | GPT-2 | AG News | 0.944 |
| Transformer | Pythia-2.8B | AG News | 0.914 |
| Transformer | Mistral-7B | XSum | 0.988 |
| Transformer | LLaMA-2-7B | AG News | 0.960 |
| **Transformer mean** | | | **0.908** |
| State-space | Mamba-2.8B | AG News | 0.971 |
| State-space | Mamba-2.8B | WikiText | 0.993 |
| Linear attention | RWKV-3B | AG News | 0.982 |
| Linear attention | RWKV-3B | XSum | 0.998 |
| **Non-transformer mean** | | | **0.957** |

## License

MIT License
