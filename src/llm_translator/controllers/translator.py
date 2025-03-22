import re

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# convert all the codes below to class and methods. AI!
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 1000
MAX_INPUT_LENGTH = round(MAX_LENGTH * 0.35)


client = LLM(model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def translate_en_to_ja(text: str, max_length: int = MAX_LENGTH) -> str:
    params = SamplingParams(max_tokens=max_length, temperature=0.9, top_p=0.95)

    prompt = f"Translate the following English text to Japanese:\n\n{text}"
    response = client.generate(prompt=prompt, sampling_params=params)
    return response[0].text.strip()


def get_token_length(text: str) -> int:
    return tokenizer([text], return_tensors="pt")["input_ids"].shape[1]


def check_max_token_length(text: str) -> int:
    paragraphs = text.split("\n")

    lengths = []
    for paragraph in paragraphs:
        length = get_token_length(paragraph)
        lengths.append(length)
    return np.max(lengths)


def split_text(text: str) -> list[str]:
    max_paragraph_length = check_max_token_length(text)

    paragraphs = []

    for paragraph in text.split("\n\n"):
        if paragraph.strip() == "":
            paragraphs.append(paragraph)
        else:
            paragraph_length = get_token_length(paragraph)
            if paragraph_length < MAX_INPUT_LENGTH:
                paragraphs.append(paragraph)
            else:
                sentences = re.split(
                    r"(?<=[.!?]) +", paragraph
                )  # use sentence splitter.
                current_paragraph = ""
                for sentence in sentences:
                    sentence_length = get_token_length(sentence)
                    if (
                        get_token_length(current_paragraph + " " + sentence)
                        <= MAX_INPUT_LENGTH
                    ):
                        current_paragraph = (current_paragraph + " " + sentence).strip()
                    else:
                        if current_paragraph:
                            paragraphs.append(current_paragraph)
                        # TODO: raise if sentence length > MAX_INPUT_LENGTH.
                        current_paragraph = sentence
                if current_paragraph:
                    paragraphs.append(current_paragraph)

    return paragraphs


def translate(text: str) -> str:
    # split text to paragraphs.
    # for paragraph in paragraphs: check the paragraph token length < 35% of max_length:
    # if longer than 35%, split paragraph to sentence.
    pass


if __name__ == "__main__":
    import sys

    input_text = sys.argv[1]
    translated_text = translate_en_to_ja(input_text)
    print(translated_text)
