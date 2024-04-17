# Czech-Bench: An evaluation framework for Czech LLMs

This repo is being created as part of a diploma thesis at FEE, CTU Prague.

## Requirements

 - Python >= 3.10

 - git-lfs

## Setup Instructions

Clone this repository:

    git lfs clone https://gitlab.com/jirkoada/czech-bench.git

Setup Python environment:

    cd czech-bench
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Usage

Inspect contents of eval_config.yml and make desired changes. Create additional config files if needed.

Then run the evaluation:

    python3 run_evaluation.py [-c path_to_custom_config.yml] [-n "Optional note to be stored in the result file"]

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

Kydlíček, [Klokánek dataset](https://huggingface.co/datasets/hynky/klokan-qa), 2023

Hendrycks et al., [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), 2021

Bowman et al., [A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326v1), 2015

Sabol and Medved’ and Horák, [Czech Question Answering with Extended SQAD v3.0 Benchmark Dataset](https://nlp.fi.muni.cz/raslan/2019/paper14-medved.pdf), 2019

Macková and Straka, [Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer](https://browse.arxiv.org/pdf/2007.01667.pdf), 2020

Rajpurkar et al., [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822), 2018

Přibáň and Steinberger, [Czech Dataset for Cross-lingual Subjectivity Classification](https://arxiv.org/abs/2204.13915), 2022

Lin et al., [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958), 2022
