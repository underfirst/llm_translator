import mistune
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# docstringはGoogle styleで書いてください. AI!
# 句点は。ではなく.を使ってください AI!
# 読点は、ではなく, (半角カンマスペース)を使ってください. AI!
# num_contextはTranslator._get_paragraphsの引数のcontext_sizeではなく__init__ methodの引数として与えてください. AI!
class Translator:
    def __init__(self, model_name: str, max_context_length: int = 1000):
        """
        Translatorクラスのコンストラクタ。使用するLLMモデルの名前と最大コンテキスト長を設定する。

        :param model_name: 使用するLLMモデルの名前。
        :param max_context_length: 翻訳時に保持するコンテキストの最大文字数。
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        """
        Markdown形式のテキスト全体を翻訳するメインメソッド。テキストを段落ごとに分割し、各段落を翻訳する。

        :param text: 翻訳対象のMarkdown形式のテキスト。
        :return: 翻訳後のMarkdown形式のテキスト。
        """
        paragraphs = self._get_paragraphs(text)
        translated_paragraphs = []
        for index, paragraph in enumerate(paragraphs):
            context = self._get_context(paragraphs, index)
            translated = self._translate_paragraph(paragraph, context)
            translated_paragraphs.append(translated)
        return "\n\n".join(translated_paragraphs)

    # contextはlist[str]という形式で, 1つのパラグラフを1つの要素とするリスト形式にしてください AI!
    def _translate_paragraph(self, target: str, context: str) -> str:
        """
        個々の段落を翻訳する内部メソッド。対象段落とその周囲のコンテキストを考慮して翻訳を行う。

        :param target: 翻訳対象の段落。
        :param context: 翻訳対象の段落の前後の段落。
        :return: 翻訳された段落。
        """
        # 特殊トークンを追加して段落の開始と終了を明示
        special_start = "<TRANSLATE_START>"
        special_end = "<TRANSLATE_END>"
        # contextの各パラグラフにもspecial_start, special_endを与えてください.
        # プロンプトは日本語で与えてください。 AI!
        prompt = f"{context}\n\n{special_start}\n{target}\n{special_end}\n\nTranslate the above paragraph into Japanese while preserving Markdown formatting."

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length,
        )
        # special_endで生成が停止するようにしてください. AI!
        # special_endが存在しない場合retryできるようにしてください. retry回数はnum_retry=3という引数で与えてください. AI!
        outputs = self.model.generate(**inputs)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

    def _get_context(self, paragraphs: list, index: int, context_size: int = 2) -> str:
        """
        指定されたインデックスの段落に対するコンテキストを取得する内部メソッド。前後の段落を結合して返す。

        :param paragraphs: 段落のリスト。
        :param index: 翻訳対象の段落のインデックス。
        :param context_size: 前後に取得する段落の数。
        :return: 翻訳対象段落の前後の段落を含むコンテキスト。
        """
        start = max(0, index - context_size)
        end = index
        context_paragraphs = paragraphs[start:end]
        return "\n\n".join(context_paragraphs)

    def _get_paragraphs(self, text: str) -> list:
        """
        Markdown形式のテキストを段落ごとに分割する内部メソッド。

        :param text: Markdown形式のテキスト。
        :return: 分割された段落のリスト。
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
        Markdownの段落の平均文字数、最大文字数などの基礎統計を取得するメソッド。

        :param text: Markdown形式のテキスト。
        :return: 統計情報を含む辞書。
        """
        paragraphs = self._get_paragraphs(text)
        lengths = [len(p) for p in paragraphs]
        # num_contextを考慮して1つのrequest (context:list[str], target_paragrah)についての平均, 最小, 最大の統計量も表示してください. AI!
        return {
            "average_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "total_paragraphs": len(paragraphs),
        }

    def get_paragraph_statistics(self, paragraph: str) -> int:
        """
        個々の段落の基礎統計を取得するメソッド。

        :param paragraph: 段落のテキスト。
        :return:
        """
        length = len(paragraph)
        return length
