# Czech MMLU dataset

The original English dataset was obtained from [Hugging Face](https://huggingface.co/datasets/cais/mmlu).

The translation was performed automatically using the [wmt21-dense-24-wide-en-x](https://huggingface.co/facebook/wmt21-dense-24-wide-en-x) model. For details, refer to the [dataset translation script](../dataset_translation.py).

The professional_law subtask is excluded from the evaluation, due to large inference expenses and low relevance to the Czech cultural setting.

### Dataset details

- Language: CS (Translated)
- Task: Knowledge-Based Question Answering
- Samples: 12505 (Test set)
- Few-shot examples: 5 (From the development set for each topic)

### Task description

MMLU consists of 57 separate subtasks, 56 of which are used here. Each subtask's examples are formatted as single-choice questions with 4 available answers. The model is asked to return a number from 1 to 4, corresponding to the chosen answer.

Accuracies for all 56 subtasks are reported, as well as averaged accuracies in 4 subtask categories (Humanities, Social Sciences, STEM, and Other) and the final average accuracy computed over all 56 accuracy values.

## License

The dataset was released under the [MIT license](LICENSE).

## References

[1] Hendrycks et al., [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), 2021
