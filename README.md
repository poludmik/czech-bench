# Czech-Bench: An evaluation framework for Czech LLMs

This repo is being created as part of a diploma theses at FEE, CTU Prague.

...

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

    python3 run_evaluation.py [-c path_to_custom_config.yml]

## References
[1]  Sabol and Medved’ and Horák, [Czech Question Answering with Extended SQAD v3.0 Benchmark Dataset](https://nlp.fi.muni.cz/raslan/2019/paper14-medved.pdf), 2019

[2] Macková and Straka, [Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer](https://browse.arxiv.org/pdf/2007.01667.pdf), 2020

[3] Ullrich, [Dataset for Automated Fact Checking in Czech
Language](https://dspace.cvut.cz/bitstream/handle/10467/95430/F3-DP-2021-Ullrich-Herbert-Thesis___Ullrich.pdf?sequence=-1&isAllowed=y), 2021
