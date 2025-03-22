import vllm

def translate_en_to_ja(text):
    client = vllm.LLMClient(model="path/to/japanese-model")
    prompt = f"Translate the following English text to Japanese:\n{text}"
    response = client.generate(prompt=prompt, max_tokens=100)
    return response[0].text.strip()

if __name__ == "__main__":
    import sys
    input_text = sys.argv[1]
    translated_text = translate_en_to_ja(input_text)
    print(translated_text)
