# Czech-Bench: An Evaluation Framework for Czech-Enabled Large Language Models

Czech-Bench is a collection of LLM benchmarks available for the Czech language. It currently includes 17 Czech benchmarks in total, accompanied by 10 English benchmark versions intended for cross-lingual performance comparison. Five datasets, ARC-Challange, ARC-Easy, GSM8K, MMLU, and TruthfulQA, were newly translated from English to Czech, and two datasets, CTKFacts and the Czech Subjectivity Dataset, were translated from Czech to English. The remaining datasets were gathered from their respective open-source repositories, which are linked and cited in each dataset's README file.

Supported models include OpenAI's and Anthropic's chat APIs, models compatible with the `AutoModelForCausalLM` and `AutoModelForSeq2SeqLM` classes of the Transformers library, and all models supported by the [Ollama](https://github.com/ollama/ollama) runtime.

This repository is being created as part of my diploma thesis at FEE, CTU Prague. It is still in active development and breaking changes may be introduced.

## Included benchmarks

All currently supported benchmarks are listed in the table below. Further details and attributions are presented in each benchmark's respective README.

| Dataset                                           | Language               | Task                       | Metrics                    | Evaluation Examples |
| ------------------------------------------------- | ---------------------- | -------------------------- | -------------------------- | ------------------: |
| [AGREE](benchmarks/agree)                         | CS (Original)          | Subject-verb agreement     | Acc                        | 627                 |
| [ANLI](benchmarks/anli)                           | CS (Translated)        | Natural Language Inference | Acc, Macro F1              | 1200                |
| [ANLI EN](benchmarks/anli_en)                     | EN (Original)          | Natural Language Inference | Acc, Macro F1              | 1200                |
| [ARC Challenge](benchmarks/arc_challenge)         | CS (Translated)        | Knowledge-Based QA         | Acc                        | 1172                |
| [ARC Challenge EN](benchmarks/arc_challenge_en)   | EN (Original)          | Knowledge-Based QA         | Acc                        | 1172                |
| [ARC Easy](benchmarks/arc_easy)                   | CS (Translated)        | Knowledge-Based QA         | Acc                        | 2376                |
| [ARC Easy EN](benchmarks/arc_easy_en)             | EN (Original)          | Knowledge-Based QA         | Acc                        | 2376                |
| [Belebele](benchmarks/belebele)                   | CS (Human translation) | Reading Comprehension / QA | Acc                        | 895                 |
| [Belebele EN](benchmarks/belebele_en)             | EN (Original)          | Reading Comprehension / QA | Acc                        | 895                 |
| [CTKFacts](benchmarks/ctkfacts)                   | CS (Original)          | Natural Language Inference | Acc, Macro F1              | 558                 |
| [CTKFacts EN](benchmarks/ctkfacts_en)             | EN (Translated)        | Natural Language Inference | Acc, Macro F1              | 558                 |
| [Czech News](benchmarks/czech_news)               | CS (Original)          | News Topic Classification  | Acc, Macro F1              | 1000                |
| [Facebook Comments](benchmarks/facebook_comments) | CS (Original)          | Sentiment Analysis         | Acc, Macro F1              | 1000                |
| [GSM8K](benchmarks/gsm8k)                         | CS (Translated)        | Mathematical inference     | EM Acc                     | 1319                |
| [GSM8K EN](benchmarks/gsm8k_en)                   | EN (Original)          | Mathematical inference     | EM Acc                     | 1319                |
| [Klokánek](benchmarks/klokanek)                   | CS (Original)          | Math/Logical Inference     | Acc                        | 808                 |
| [Mall Reviews](benchmarks/mall_reviews)           | CS (Original)          | Sentiment Analysis         | Acc, Macro F1              | 3000                |
| [MMLU](benchmarks/mmlu)                           | CS (Translated)        | Knowledge-Based QA         | Acc                        | 12505               |
| [MMLU EN](benchmarks/mmlu_en)                     | EN (Original)          | Knowledge-Based QA         | Acc                        | 12505               |
| [SNLI](benchmarks/snli)                           | CS (Translated)        | Natural Language Inference | Acc, Macro F1              | 10000               |
| [SNLI EN](benchmarks/snli_en)                     | EN (Original)          | Natural Language Inference | Acc, Macro F1              | 10000               |
| [SQAD](benchmarks/sqad)                           | CS (Original)          | Reading Comprehension / QA | EM Acc, BoW F1             | 843                 |
| [SQuAD](benchmarks/squad)                         | CS (Translated)        | Reading Comprehension / QA | EM Acc, F1, No-Ans Acc, F1 | 4000                |
| [Subjectivity](benchmarks/subjectivity)           | CS (Original)          | Subjectivity Analysis      | Acc, Macro F1              | 2000                |
| [Subjectivity EN](benchmarks/subjectivity_en)     | EN (Translated)        | Subjectivity Analysis      | Acc, Macro F1              | 2000                |
| [TruthfulQA](benchmarks/truthfulqa)               | CS (Translated)        | Knowledge-Based QA         | Acc                        | 813                 |
| [TruthfulQA EN](benchmarks/truthfulqa_en)         | EN (Original)          | Knowledge-Based QA         | Acc                        | 813                 |

## Usage Instructions

### Requirements:

 - Python >= 3.10

 - git-lfs

### Setup:

Clone this repository:

    git lfs clone https://gitlab.com/jirkoada/czech-bench.git

Setup Python environment:

    cd czech-bench
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Evaluation:

Inspect the contents of [*eval_config.yml*](eval_config.yml) and make desired changes. Consult the model integrations [README](models) for further reference. Create additional config files if needed.

Then run the evaluation:

    python3 run_evaluation.py [-c path_to_custom_config.yml] [-n "Optional note to be stored in the result file"]

### Limitations: 

Batch inference is currently not supported, so all evaluation examples are processed sequentially. Any help with introducing batch inference, or possibly even porting the included datasets into the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework would be greatly appreciated.

## Interim results

The table below provides a performance comparison of two popular commercial models, OpenAI's GPT-3.5 Turbo (version 0125) and Anthropic's Claude 3 Haiku (version 20240307). Evaluation of available multilingual open-source models is currently underway. Details about individual datasets and reported metrics can be found in each benchmark's respective README.

| Dataset           | Metrics                        | GPT 3.5 turbo              | Claude 3 Haiku             |
| ----------------- | ------------------------------ | -------------------------: | -------------------------: |
| AGREE             | Acc                            | 46.7                       | 65.7                       |
| ANLI              | Acc, Macro F1                  | 44.67, 41.93               | 51.50, 50.75               |
| ANLI EN           | Acc, Macro F1                  | 44.25, 40.58               | 55.34, 54.16               |
| ARC Challenge     | Acc                            | 73.1                       | 76.8                       |
| ARC Challenge EN  | Acc                            | 82.9                       | 77.6                       |
| ARC Easy          | Acc                            | 85.8                       | 85.3                       |
| ARC Easy EN       | Acc                            | 93.1                       | 89.1                       |
| Belebele          | Acc                            | 80.3                       | 88.2                       |
| Belebele EN       | Acc                            | 87.0                       | 91.0                       |
| CTKFacts          | Acc, Macro F1                  | 61.83, 47.71               | 69.57, 62.03               |
| CTKFacts EN       | Acc, Macro F1                  | 67.56, 63.23               | 68.06, 62.22               |
| Czech News        | Acc, Macro F1                  | 78.9, 78.45                | 81.3, 81.31                |
| Facebook Comments | Acc, Macro F1                  | 71.50, 69.02               | 75.8, 74.09                |
| GSM8K             | EM Acc                         | 64.2                       | 78.6                       |
| GSM8K EN          | EM Acc                         | 83.1                       | 89.0                       |
| Klokánek          | Acc                            | 29.3                       | 24.5                       |
| Mall Reviews      | Acc, Macro F1                  | 59.76, 55.42               | 57.67, 55.23               |
| MMLU              | Acc                            | 58.0                       | 67.3                       |
| MMLU EN           | Acc                            | 64.9                       | 73.0                       |
| SNLI              | Acc, Macro F1                  | 61.8, 51.49                | 71.66, 70.48               |
| SNLI EN           | Acc, Macro F1                  | 60.57, 43.32               | 72.74, 53.78               |
| SQAD              | EM Acc, BoW F1                 | 66.19, 83.47               | 59.79, 76.25               |
| SQuAD             | EM Acc, BoW F1, No-Ans Acc, F1 | 37.30, 42.98, 52.42, 44.21 | 36.25, 44.67, 60.27, 56.38 |
| Subjectivity      | Acc, Macro F1                  | 80.15, 80.15               | 81.5, 81.21                |
| Subjectivity EN   | Acc, Macro F1                  | 86.8, 86.79                | 86.6, 86.59                |
| TruthfulQA        | Acc                            | 53.5                       | 65.8                       |
| TruthfulQA EN     | Acc                            | 58.5                       | 70.8                       |

## References

Baisa, [Byte Level Language Models](https://is.muni.cz/th/139654/fi_d/), 2016

Ullrich, [Dataset for Automated Fact Checking in Czech
Language](https://dspace.cvut.cz/bitstream/handle/10467/95430/F3-DP-2021-Ullrich-Herbert-Thesis___Ullrich.pdf?sequence=-1&isAllowed=y), 2021

Nie et al., [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599v2), 2020

Clark et al., [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457), 2018

Bandarkar et al., [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884), 2023

Ullrich et al., [CsFEVER and CTKFacts: Acquiring Czech data for fact verification](https://arxiv.org/abs/2201.11115), 2022

Kydlíček et al., [A Dataset and Strong Baselines for Classification of Czech News Texts](https://arxiv.org/abs/2307.10666), 2023

Habernal et al., [Sentiment Analysis in Czech Social Media Using Supervised Machine Learning](https://aclanthology.org/W13-1609/), 2013

Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), 2021

Kydlíček, [Klokánek dataset](https://huggingface.co/datasets/hynky/klokan-qa), 2023

Hendrycks et al., [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), 2021

Bowman et al., [A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326v1), 2015

Sabol and Medved’ and Horák, [Czech Question Answering with Extended SQAD v3.0 Benchmark Dataset](https://nlp.fi.muni.cz/raslan/2019/paper14-medved.pdf), 2019

Macková and Straka, [Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer](https://browse.arxiv.org/pdf/2007.01667.pdf), 2020

Rajpurkar et al., [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822), 2018

Přibáň and Steinberger, [Czech Dataset for Cross-lingual Subjectivity Classification](https://arxiv.org/abs/2204.13915), 2022

Lin et al., [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958), 2022
