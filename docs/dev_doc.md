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


<!-- AI: 以下はLLMを用いたMarkdownの翻訳のためのAPI定義です. 実装に必要な詳細情報を追加してください. AI! -->

モジュールはTranslatorクラスのメソッドとして実装する.
Translatorクラスは以下のメソッドを持つ.

- `Translator.__init__(self, model_name:str, max_context_length:int)`: コンストラクタ.
- `Translator.translate(self, text: str) -> str`: Markdown全文を翻訳するメインモジュール.
- `Translator._translate_paragraph(self, target, context) -> str`: 段落を翻訳するモジュール.
  - target: 翻訳対象の段落.
  - context: 翻訳対象の段落の前後の段落.
- `Translator._get_context(self, paragraphs, index) -> str`: 翻訳対象の段落の前後の段落を取得するモジュール.
  - paragraphs: 段落のリスト.
  - index: 翻訳対象の段落のインデックス.
- `Translator._get_paragraphs(self, text) -> list`: Markdownを段落ごとに分割するモジュール.
  - text: Markdown形式のテキスト.
- `Translarator.get_statistics(self, text) -> None`: Markdownの段落の平均文字数, 最大文字数などの基礎統計を確認するためのモジュール.
  - text: Markdown形式のテキスト.
- `Translator.get_paragraph_statistics(self, paragraph:str)`: 段落の基礎統計を取得するモジュール.
  - paragraph: 段落のテキスト.

