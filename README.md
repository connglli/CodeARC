# CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis (COLM'25)

[![COLM 2025](https://img.shields.io/badge/Accepted-COLM%2725-blue)](https://colmweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.12466-b31b1b.svg)](https://arxiv.org/abs/2503.23145)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange.svg)](https://huggingface.co/datasets/anjiangwei/CodeARC-Problems)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-10%20Input--Output%20Examples-orange.svg)](https://huggingface.co/datasets/anjiangwei/CodeARC-Invocations)

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-%20Model%20Annotated-orange.svg)](https://huggingface.co/LLM4Code/CodeARC_annotated_llama3.1)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-%20Model%20Anonymized-orange.svg)](https://huggingface.co/LLM4Code/CodeARC_anonymous_llama3.1)




## Quick Start

### Setting Up the Environment

1. **Create and activate a Conda environment**:

   ```bash
   conda create -y -n CodeARC python=3.10.12
   conda activate CodeARC
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set API keys**:
   Ensure you have valid API keys for the required services:

   ```bash
   export OPENAI_API_KEY=<your_openai_api_key>
   export ANTHROPIC_API_KEY=<your_anthropic_api_key>
   export TOGETHER_API_KEY=<your_together_api_key>
   ```


## Running Main Evaluation


   ```python
   python3 run.py --model_name openai/gpt-4o-mini --total_idx 20
   ```
   We support OpenAI models (e.g., `openai/gpt-4o`), Anthropic models (e.g., `anthropic/claude-3-7-sonnet-20250219`), and models served by Together AI (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`). For testing purposes, you can pass `--total_idx 20` to limit evaluation to 20 problems instead of the full dataset (1114 problems). See run.py for additional configuration options.
   
   To summarize results:
   ```python
   python3 src/compute_metrics.py
   ```

---

## HuggingFace Dataset

The CodeARC datasets are hosted on HuggingFace:

- **Problems Dataset**: [anjiangwei/CodeARC-Problems](https://huggingface.co/datasets/anjiangwei/CodeARC-Problems)
- **Invocations Dataset**: [anjiangwei/CodeARC-Invocations](https://huggingface.co/datasets/anjiangwei/CodeARC-Invocations)

### Setting up HuggingFace Account

1. **Obtain an access token**:
   - Go to [HuggingFace Tokens](https://huggingface.co/settings/tokens) and generate a token with `read` or `write` permissions.

2. **Login using the token**:

   **Option A**: Use the command line:

   ```bash
   huggingface-cli login
   huggingface-cli whoami
   ```

   **Option B**: Add the token to the environment variable:

   ```plaintext
   export HF_TOKEN=<your_huggingface_token>
   ```

### Accessing Datasets via the HuggingFace `datasets` Library

You can directly load the datasets using the HuggingFace `datasets` library:

```python
from datasets import load_dataset

# Define dataset paths
hf_problems_path = "anjiangwei/CodeARC-Problems"
hf_invocations_path = "anjiangwei/CodeARC-Invocations"

# Load datasets
problems_dataset = load_dataset(hf_problems_path)
invocations_dataset = load_dataset(hf_invocations_path)

# Example: Access the first training sample
print(problems_dataset["train"][0])
print(invocations_dataset["train"][0])
```

---

## Citation

If our research inspires you, please cite our paper:

```bibtex
@inproceedings{wei2025codearc,
  title={Code{ARC}: Benchmarking Reasoning Capabilities of {LLM} Agents for Inductive Program Synthesis},
  author={Anjiang Wei and Tarun Suresh and Jiannan Cao and Naveen Kannan and Yuheng Wu and Kai Yan and Thiago S. F. X. Teixeira and Ke Wang and Alex Aiken},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://openreview.net/forum?id=Q5pVZCrrKr}
}
```

---

## License

This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).
