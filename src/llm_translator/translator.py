import mistune
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self, model_name: str, max_context_length: int = 1000, num_context: int = 2):
        """
        Translator class constructor. Sets the name of the LLM model to use and the maximum context length.

        :param model_name: Name of the LLM model to use.
        :param max_context_length: Maximum number of characters to maintain as context during translation.
        :param num_context: Number of surrounding paragraphs to include as context for each translation request.
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.num_context = num_context
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        """
        Main method to translate an entire Markdown-formatted text. Splits the text into paragraphs and translates each one.

        :param text: Markdown-formatted text to translate.
        :return: Translated Markdown-formatted text.
        """
        paragraphs = self._get_paragraphs(text)
        translated_paragraphs = []
        for index, paragraph in enumerate(paragraphs):
            context = self._get_context(paragraphs, index)
            translated = self._translate_paragraph(paragraph, context)
            translated_paragraphs.append(translated)
        return "\n\n".join(translated_paragraphs)

    def _translate_paragraph(self, target: str, context: list[str], num_retry: int = 3) -> str:
        """
        Internal method to translate an individual paragraph, considering its surrounding context.

        :param target: Paragraph to translate.
        :param context: List of surrounding paragraphs providing context.
        :param num_retry: Number of retry attempts if the translation does not contain the special end token.
        :return: Translated paragraph.
        """
        special_start = "<TRANSLATE_START>"
        special_end = "<TRANSLATE_END>"
        formatted_context = "\n\n".join([f"{special_start}\n{para}\n{special_end}" for para in context])
        prompt = (
            f"{formatted_context}\n\n{special_start}\n{target}\n{special_end}\n\n"
            "上記の段落をMarkdownのフォーマットを保持しつつ日本語に翻訳してください."
        )

        eos_token_id = self.tokenizer.encode(special_end, add_special_tokens=False)[-1]

        for attempt in range(num_retry):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length,
            )
            outputs = self.model.generate(
                **inputs,
                eos_token_id=eos_token_id
            )
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if special_end in translation:
                translation = translation.split(special_end)[0] + special_end
                return translation
            else:
                if attempt < num_retry - 1:
                    continue
                else:
                    raise ValueError("Translation did not contain the special end token after multiple retries.")
        return translation

    def _get_context(self, paragraphs: list, index: int) -> list[str]:
        """
        Retrieves the context for a specified paragraph index by collecting surrounding paragraphs.

        :param paragraphs: List of all paragraphs.
        :param index: Index of the target paragraph.
        :return: List of surrounding paragraphs to be used as context.
        """
        start = max(0, index - self.num_context)
        end = index
        context_paragraphs = paragraphs[start:end]
        return context_paragraphs

    def _get_paragraphs(self, text: str) -> list:
        """
        Splits Markdown-formatted text into individual paragraphs.

        :param text: Markdown-formatted text.
        :return: List of paragraphs.
        """
        markdown = mistune.create_markdown(renderer=None)
        parsed = markdown(text)
        paragraphs = []
        for block in parsed:
            if block["type"] == "paragraph":
                paragraph = "".join(child["raw"] for child in block["children"])
                paragraphs.append(paragraph)
        return paragraphs

    def get_statistics(self, text: str) -> dict:
        """
        Retrieves basic statistics of the Markdown paragraphs, including average, maximum, and minimum lengths,
        as well as statistics related to the translation requests considering the number of context paragraphs.

        :param text: Markdown-formatted text.
        :return: Dictionary containing statistical information.
        """
        paragraphs = self._get_paragraphs(text)
        lengths = [len(p) for p in paragraphs]
        total_requests = len(paragraphs)
        context_counts = [len(self._get_context(paragraphs, idx)) for idx in range(total_requests)]
        avg_context = sum(context_counts) / total_requests if total_requests else 0
        max_context = max(context_counts) if total_requests else 0
        min_context = min(context_counts) if total_requests else 0
        return {
            "average_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "total_paragraphs": len(paragraphs),
            "average_context_per_request": avg_context,
            "max_context_per_request": max_context,
            "min_context_per_request": min_context,
        }

    def get_paragraph_statistics(self, paragraph: str) -> dict:
        """
        Retrieves basic statistics of an individual paragraph.

        :param paragraph: Text of the paragraph.
        :return: Dictionary containing statistical information of the paragraph.
        """
        length = len(paragraph)
        return {
            "length": length
        }
