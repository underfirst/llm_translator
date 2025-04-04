# LLMを用いたmd翻訳

## 概要

LLMを用いてMarkdown形式のドキュメントを翻訳するためのスクリプトを作成する.
短文の翻訳としては, 単純に, 「以下のテキストを日本語に翻訳してください。」のようなプロンプトを与えるだけで良いが, Markdown形式のドキュメントを翻訳する場合は, 以下の課題が生じる.

1. Markdown形式でない形式で翻訳してしまう.
2. 長い文章を入力とすることでモデルが正常に翻訳しない.
  - モデルのトークン数制限に引っかかる.
  - モデルが勝手に内容を要約してしまう.

上記課題に対して次のように対処する.

まず, Markdown 形式のドキュメントを, Markdownパーサを使って, パラグラフごとに分割する.
Markdownパーサとして, mistuneを用いる.
mistuneは数式などの拡張表現も対応している.
mistuneのパーサは次のように構文木を取得できる.


```py
import mistune


text = 'hello **world**'

markdown = mistune.create_markdown(renderer=None)
markdown(text)
# ==>
[
    {
        'type': 'paragraph',
        'children': [
            {'type': 'text', 'raw': 'hello '},
            {'type': 'strong', 'children': [{'type': 'text', 'raw': 'world'}]}
        ]
    }
]
```

このようにして, markdownの段落を取得したあと, 段落ごとにテキストを翻訳する.
翻訳の際に, Markdownの表現を維持するようにPromptで指示を与える.
また, 文脈を維持するため, 翻訳対象の段落の直前n個の段落の原文および, その翻訳を与える.
さらに, 対象の段落が翻訳されたことを確認するために, 段落の開始および終了を表す特殊トークンを各段落の開始/終了部分に付与する.


## API

モジュールはTranslatorクラスのメソッドとして実装する.
Translatorクラスは以下のメソッドを持つ.

- `Translator.__init__(self, model_name: str, max_context_length: int)`
  - **説明**: Translatorクラスのコンストラクタ。使用するLLMモデルの名前と最大コンテキスト長を設定する。
  - **パラメータ**:
    - `model_name` (str): 使用するLLMモデルの名前。
    - `max_context_length` (int): 翻訳時に保持するコンテキストの最大文字数。

- `Translator.translate(self, text: str) -> str`
  - **説明**: Markdown形式のテキスト全体を翻訳するメインメソッド。テキストを段落ごとに分割し、各段落を翻訳する。
  - **パラメータ**:
    - `text` (str): 翻訳対象のMarkdown形式のテキスト。
  - **戻り値**:
    - `str`: 翻訳後のMarkdown形式のテキスト。

- `Translator._translate_paragraph(self, target: str, context: str) -> str`
  - **説明**: 個々の段落を翻訳する内部メソッド。対象段落とその周囲のコンテキストを考慮して翻訳を行う。
  - **パラメータ**:
    - `target` (str): 翻訳対象の段落。
    - `context` (str): 翻訳対象の段落の前後の段落。
  - **戻り値**:
    - `str`: 翻訳された段落。

- `Translator._get_context(self, paragraphs: list, index: int) -> str`
  - **説明**: 指定されたインデックスの段落に対するコンテキストを取得する内部メソッド。前後の段落を結合して返す。
  - **パラメータ**:
    - `paragraphs` (list): 段落のリスト。
    - `index` (int): 翻訳対象の段落のインデックス。
  - **戻り値**:
    - `str`: 翻訳対象段落の前後の段落を含むコンテキスト。

- `Translator._get_paragraphs(self, text: str) -> list`
  - **説明**: Markdown形式のテキストを段落ごとに分割する内部メソッド。
  - **パラメータ**:
    - `text` (str): Markdown形式のテキスト。
  - **戻り値**:
    - `list`: 分割された段落のリスト。

- `Translator.get_statistics(self, text: str) -> dict`
  - **説明**: Markdownの段落の平均文字数、最大文字数などの基礎統計を取得するメソッド。
  - **パラメータ**:
    - `text` (str): Markdown形式のテキスト。
  - **戻り値**:
    - `dict`: 統計情報を含む辞書。

- `Translator.get_paragraph_statistics(self, paragraph: str) -> dict`
  - **説明**: 個々の段落の基礎統計を取得するメソッド。
  - **パラメータ**:
    - `paragraph` (str): 段落のテキスト。
  - **戻り値**:
    - `dict`: 段落の統計情報を含む辞書。
