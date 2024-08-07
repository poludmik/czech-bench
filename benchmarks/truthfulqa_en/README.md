# TruthfulQA (Original English) dataset

The dataset was obtained from [Hugging Face](https://huggingface.co/datasets/truthful_qa). Only the single-choice (mc1) variant of the dataset is used, and the ordering of proposed answers is randomized.

### Dataset details

- Language: EN
- Task: Knowledge-Based Question Answering
- Samples: 817 (Validation set)
- Few-shot examples: 5 (From validation set, excluded from evaluation)

### Task description

The model is given a question targeted at common misinformation and a variable-sized set of possible answers, only one of which is correct. It is expected to return the number corresponding to the chosen answer.

The reported accuracy metric represents the percentage of correctly selected answers.

## License

The dataset was released under the [Apache License 2.0](LICENSE).

## References

[1] Lin et al., [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958), 2022
