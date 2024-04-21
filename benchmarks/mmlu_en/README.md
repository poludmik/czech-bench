# MMLU (Original English) dataset

The dataset was obtained from [Hugging Face](https://huggingface.co/datasets/cais/mmlu).

The professional_law subtask is excluded from the evaluation in agreement with the Czech version of the dataset.

### Dataset details

- Language: EN
- Task: Knowledge-Based Question Answering
- Samples: 12505 (Test set)
- Few-shot examples: 5 (From the development set for each topic)

### Task description

MMLU consists of 57 separate subtasks, 56 of which are used here. Each subtask's examples are formatted as single-choice questions with 4 available answers. The model is asked to return a number from 1 to 4, corresponding to the chosen answer.

Accuracies for all 56 subtasks are reported, as well as averaged accuracies in 4 subtask categories (Humanities, Social Sciences, STEM, and Other) and the final average accuracy computed over all 56 accuracy values.

## References

[1] Hendrycks et al., [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), 2021
