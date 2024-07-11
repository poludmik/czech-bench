# Czech-Bench: An Evaluation Framework for Czech-Enabled Large Language Models

Czech-Bench is a collection of LLM benchmarks available for the Czech language. It currently includes 17 Czech benchmarks in total, accompanied by 10 English benchmark versions intended for cross-lingual performance comparison. Five datasets, ARC-Challange, ARC-Easy, GSM8K, MMLU, and TruthfulQA, were newly translated from English into Czech, and two datasets, CTKFacts and the Czech Subjectivity Dataset, were translated from Czech into English. The remaining datasets were gathered from their respective open-source repositories, which are linked and cited in each dataset's README file.

Supported models include OpenAI's and Anthropic's chat APIs, models compatible with the `AutoModelForCausalLM` and `AutoModelForSeq2SeqLM` classes of the Transformers library, and all models supported by the [Ollama](https://github.com/ollama/ollama) runtime.

This repository was created as part of my [master's thesis](https://dspace.cvut.cz/handle/10467/115227) at FEE, CTU Prague.

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
| [MMLU](benchmarks/mmlu)                           | CS (Translated)        | Knowledge-Based QA         | Acc                        | 12508               |
| [MMLU EN](benchmarks/mmlu_en)                     | EN (Original)          | Knowledge-Based QA         | Acc                        | 12508               |
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

 - [git-lfs](https://git-lfs.com/)

### Setup:

Clone this repository:

    git lfs clone https://gitlab.com/jirkoada/czech-bench.git

Set up Python environment:

    cd czech-bench
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Evaluation:

Inspect the contents of [*eval_config.yml*](eval_config.yml) and make desired changes. Consult the model integrations [README](models) for further reference. Create additional config files if needed.

Then run the evaluation:

    python3 run_evaluation.py [-c path_to_custom_config.yml] [-n "Optional note to be stored in the result file"]

### Limitations: 

Batch inference is currently not supported, so all evaluation examples are processed sequentially. A team effort at integrating Czech-Bench into the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework is currently underway, aiming to levarage its advanced feature set. 

## Evaluation results

Please see the dedicated [eval_results.md](eval_results.md) file for a comprehensive evaluation report and result interpretations.

### TL;DR: 

Claude 3 Haiku outperforms GPT-3.5 Turbo in the majority of Czech benchmarks while being offered at a significantly lower price. This makes it a clear choice for potential adopters of entry-level commercial LLM APIs. Claude 3 Sonnet, on the other hand, fails to deliver any convincing performance gains over its cheaper variant and is thus not recommended considering its substantially higher price. GPT-4o is a solid mid-tier offering, beating its more expensive predecessor, the GPT-4 Turbo. Claude 3 Opus offers the highest grammatical competence in the Czech language for a significant price premium.

Llama 3 8B Instruct achieves a decisive victory among all evaluated open-source models, matching the performance of entry-level commercial offerings on several occasions. It is an excellent candidate for further fine-tuning efforts and for potential open-source LLM adopters. The larger Llama 3 70B model has not yet been evaluated.

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

Straková et al., [Open-Source Tools for Morphology, Lemmatization, POS Tagging and Named Entity Recognition](https://aclanthology.org/P14-5003), 2014

Vidra et al., [DeriNet 2.0: Towards an All-in-One Word-Formation Resource](https://aclanthology.org/W19-8510), 2019

Jirkovský, [Benchmarking Techniques for Evaluation of Large Language Models](https://dspace.cvut.cz/handle/10467/115227), 2024
