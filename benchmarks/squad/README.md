# Czech SQuAD Dataset 

This dataset was obtained from [LINDAT](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3249). See the [original README](README_ORIGINAL.md) for further details.

After loading the whole dataset of 10845 samples, only the first 4000 samples are selected for the evaluation. This is to limit inference costs and avoid breaching any tokens-per-day limits imposed by some API providers. The resulting subset contains 1825 answerable and 2175 unanswerable questions.

### Dataset details

- Language: CS (Translated)
- Task: Question Answering / Reading Comprehension
- Samples: 4000 (Development set)
- Few-shot examples: 5 (From training set)

### Task description

The model is presented with a source passage and a question, that is not guaranteed to be answerable based on the passage contents. The model is expected to either correctly extract the answer to the question from the source passage, or return a single '-' token if the answer is not present.

The exact match accuracy and ROUGE-1-like F1 score obtained using the [SQuAD v2 metric](https://huggingface.co/spaces/evaluate-metric/squad_v2) from Hugging Face are reported, as well as the accuracy and macro-averaged F1 score in the binary classification problem of detecting unanswerable questions.

## References

[1] Macková and Straka, [Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer](https://arxiv.org/abs/2007.01667), 2020

[2] Rajpurkar et al., [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822), 2018
