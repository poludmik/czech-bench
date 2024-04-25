import re
import torch
from torch import cuda
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from datasets import load_dataset


PENALTY = 5.0  # Model repetition penalty, 5 seems to be a reasonable value
TO_EN = True  # Translate to English, if False, translate to Czech


math_pattern = re.compile(r'^[\d+\-\−*/()%,. ]+$') # Regular expression for math expressions
bool_pairs = {"true, false": "pravda, nepravda", "true, true": "pravda, pravda", "false, false": "nepravda, nepravda", "false, true": "nepravda, pravda"}


def translate(model, tokenizer, texts, device):
    '''
    Translates a batch of short texts (up to 200 tokens each)
    '''
    if TO_EN:
        tokenized_texts = tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
        generated_tokens = model.generate(**tokenized_texts.to(device))
    else:
        tokenized_texts = tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
        generated_tokens = model.generate(**tokenized_texts.to(device), forced_bos_token_id=tokenizer.get_lang_id("cs"))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(result)
    return result


def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def translate_by_sentences(text, batch_size, model, tokenizer, device):
    """
    Splits the text into sentences and translates them separately, to comply with the model's input length limit
    """
    if text.isnumeric() or re.match(math_pattern, text):  # Ignore isolated math expressions
        print(text)
        return text
    if not TO_EN:
        if text.lower() == "i only":  # Handles correct translation of the answer "I only" (alternative to "I and II", "III only", etc.)
            print("Pouze I")
            return "Pouze I"
        if text.lower() in bool_pairs.keys():  # Direct translation of boolean pair answers
            print(bool_pairs[text.lower()])
            return bool_pairs[text.lower()]
    
    text = text.strip()
    if text == "":
        return ""

    sentences = [str(i).strip() for i in nlp(text).sents]
    translated_sentences = []
    for batch in batch_data(sentences, batch_size):
        translated_sentences += translate(model, tokenizer, batch, device)
    
    output =  " ".join(translated_sentences)

    if not TO_EN and len(output.split()) == 1 and "ová" in output:  # Mitigate unexpected feminine surname forms
        print(f"{output} -> {text}")
        return text
    return output


def translate_arc_example(example):
    example['question'] = translate_by_sentences(example['question'], 8, model, tokenizer, device)
    example['choices']['text'] = [translate_by_sentences(choice, 1, model, tokenizer, device) for choice in example['choices']['text']]
    return example


def translate_mmlu_example(example):
    example['question'] = translate_by_sentences(example['question'], 8, model, tokenizer, device)
    example['choices'] = [translate_by_sentences(choice, 1, model, tokenizer, device) for choice in example['choices']]
    return example


def translate_truthfulqa_example(example):
    example['question'] = translate_by_sentences(example['question'], 8, model, tokenizer, device)
    example['mc1_targets']['choices'] = [translate_by_sentences(choice, 1, model, tokenizer, device) for choice in example['mc1_targets']['choices']]
    example['mc2_targets']['choices'] = [translate_by_sentences(choice, 1, model, tokenizer, device) for choice in example['mc2_targets']['choices']]
    return example


def translate_gsm8k_example(example):
    example['question'] = translate_by_sentences(example['question'], 8, model, tokenizer, device)

    match = re.search(r'^(.*)####', example['answer'], re.DOTALL)  # Extract the thought process from the answer
    if match:
        clear_thoughts = re.sub(r'<<.*?>>', '', match.group(1))  # Remove calculator annotations
        example['thoughts'] = translate_by_sentences(clear_thoughts, 8, model, tokenizer, device)
    else:
        example['thoughts'] = ""

    match = re.search(r'#### ([\d,-]+)$', example['answer'])  # Extract the final numerical answer
    ans = match.group(1)
    ans = ans.replace(',', '')
    example['answer'] = int(ans)
    return example


def translate_subjectivity_example(example):
    example['text'] = translate_by_sentences(example['text'], 8, model, tokenizer, device)
    return example


def translate_ctkfacts_example(example):
    example['evidence'] = translate_by_sentences(example['evidence'], 8, model, tokenizer, device)
    example['claim'] = translate_by_sentences(example['claim'], 1, model, tokenizer, device)
    return example


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')  # Works sufficiently in Czech as well
    print(f"Repetition penalty: {PENALTY}")

    data = load_dataset("pauli31/czech-subjectivity-dataset", split="train")  # Load the dataset
    print(len(data))

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(device)

    if TO_EN:
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-x-en", torch_dtype=torch.bfloat16, device_map='auto', repetition_penalty=PENALTY)
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-x-en")
        tokenizer.src_lang = "cs"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x", torch_dtype=torch.bfloat16, device_map='auto', repetition_penalty=PENALTY)
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

    translated_data = data.map(translate_subjectivity_example)  # Select the correct map function

    translated_data.save_to_disk(f"./translated/subjectivity/train_en")  # Save the translated dataset
