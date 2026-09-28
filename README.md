# Detoxifying Dialogue Summaries with FLAN-T5

This project explores how reinforcement learning can help a language model generate **less toxic dialogue summaries** while keeping the summaries relevant to the original conversation.

The notebook starts with a FLAN-T5 model adapted for dialogue summarization, then uses Proximal Policy Optimization (PPO) and a hate-speech reward model to encourage safer generations. It uses **LoRA**, a parameter-efficient fine-tuning technique, to update a small portion of the model’s parameters.

> This project demonstrates a research workflow. A lower toxicity score from the reward model does not guarantee that every generated summary is harmless, unbiased, or factually correct.

## Project goals

- Generate concise summaries of dialogues.
- Measure the toxicity of generated summaries with a pretrained classifier.
- Fine-tune the summarization model with PPO to reduce the classifier’s hate-speech signal.
- Compare model behavior before and after fine-tuning.

## How it works

The workflow in the notebook has these stages:

1. **Load the data and summarization model.** The notebook uses the [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum) and a FLAN-T5 model adapted for summarization.
2. **Prepare the policy model.** LoRA adapters are attached to FLAN-T5 so PPO can update a smaller set of parameters.
3. **Prepare the reward model.** The [RoBERTa hate-speech classifier](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) assigns scores for the `nothate` and `hate` classes.
4. **Generate candidate summaries.** The policy model summarizes dialogue examples from DialogSum.
5. **Score generated summaries.** The classifier’s `nothate` score is used as a positive reward signal.
6. **Update the policy with PPO.** PPO adjusts the summarization model to improve its reward. A reference model and KL-divergence penalty help keep the updated model from drifting too far from its starting behavior.
7. **Evaluate the results.** The notebook compares toxicity scores and examples from before and after fine-tuning.

### Important distinction

The reward model is a **proxy for human feedback**, not a complete definition of safe language. Optimizing its score can reduce the signal it detects while still producing poor, biased, or inaccurate summaries. Review output quality as well as toxicity scores.

## Repository contents

- `fine_tuning_FLAN_T5-main/fine_tune_model_to_detoxify_summaries.ipynb` — the end-to-end notebook for data preparation, model setup, PPO fine-tuning, and evaluation.

## Requirements

Run the notebook in a Python environment with a compatible GPU runtime. The notebook installs or uses packages from the Hugging Face and PyTorch ecosystem, including:

- PyTorch
- Transformers
- Datasets
- Evaluate
- PEFT
- TRL
- Accelerate
- NumPy
- pandas
- tqdm

Package APIs change over time, especially in `trl` and `transformers`. If you run into compatibility errors, use package versions that work together and update the notebook’s PPO setup accordingly.

## Run the notebook

1. Clone or download this repository.
2. Open the notebook in Jupyter or Google Colab.
3. Select a GPU runtime with enough memory for the loaded models.
4. Run the notebook cells in order.
5. Check that all required model checkpoints are available before starting training.

The notebook loads pretrained models and datasets from Hugging Face, so the first run requires an internet connection. PPO training can take significant time and compute.

### Checkpoint requirement

The notebook expects a summarization adapter checkpoint in a local directory named:

```text
./peft-dialogue-summary-checkpoint-from-s3/
```

Make sure this checkpoint is available in the notebook’s runtime before running the cells that load it. If you use a different location, update the checkpoint path in the notebook.

## Models and data

- **Base model:** FLAN-T5, a text-to-text instruction-tuned model.
- **Summarization adapter:** A LoRA/PEFT adapter used to provide the initial dialogue-summarization behavior.
- **Dataset:** [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum), containing dialogues and human-written summaries.
- **Reward model:** [RoBERTa hate-speech classifier](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target), which predicts `nothate` or `hate`.

Refer to the notebook for the exact model configuration, preprocessing, PPO parameters, and evaluation code.

## Results and evaluation

The notebook evaluates model outputs before and after PPO fine-tuning using the toxicity classifier and qualitative examples.

When reporting results, include the actual values produced by your run—for example, the mean toxicity or hate score before and after fine-tuning. Results depend on the checkpoint, dataset subset, generation settings, package versions, and random seed, so they should not be treated as universal performance claims.

A useful evaluation should also consider:

- Whether the generated summary preserves the dialogue’s meaning.
- Whether important details are omitted or invented.
- Whether the language remains natural and readable.
- Whether toxicity changes on examples outside the training subset.

## Limitations

- The reward model can make classification mistakes and may reflect biases in its training data.
- PPO can optimize for the reward model’s score without improving every aspect of summary quality.
- Reduced hate-speech scores do not establish that outputs are safe or fair.
- Results from a limited dataset or notebook run may not generalize to other topics, languages, or models.
- Fine-tuning requires substantial compute and compatible versions of the supporting libraries.

## Reproducibility

For reproducible results, record:

- Python, PyTorch, Transformers, PEFT, and TRL versions.
- The exact base-model and adapter checkpoint identifiers.
- PPO configuration and generation settings.
- Dataset split or subset, and random seeds.
- Hardware and runtime details.
- Before-and-after evaluation results.

## License and attribution

Add the appropriate license for this repository and follow the licenses and usage terms for the models, datasets, and libraries used. Cite the original sources when reusing their models, datasets, or notebook material.
