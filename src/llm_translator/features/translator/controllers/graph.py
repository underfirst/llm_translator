from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from tqdm import tqdm
from typer import Typer


class TechnicalTerm(BaseModel):
    original: str = ""
    translated: str = ""


class Passage(BaseModel):
    original: str = ""
    translated: str = ""
    is_meta_text: bool = False


class TranslatorState(BaseModel):
    terms: list[TechnicalTerm] = []
    original_text: str
    passages: list[Passage] = []
    callbacks: list = []


llm = ChatOpenAI(model="gpt-4o")


def parse_text(state: TranslatorState) -> TranslatorState:
    # TODO: regexを用いて, 2つ以上連続した改行でoriginal_textを分割.
    passages = []
    for text in state.original_text.split("\n\n"):
        passages.append(Passage(original=text))
    state.passages = passages
    return state


def detect_meta_text(state: TranslatorState) -> TranslatorState:
    class MetaTextCheckResponse(BaseModel):
        """
        与えたテキストが, 英語論文のメタテキストかどうかを判断するための応答モデル.
        メタテキストとは, 本文やタイトル以外のテキストのことを指す.
        具体的には, 画像のキャプション, 著者名, 数式, 図のリンクや, テーブル, モデルの出力例, 数式, 参考文献などを含む.
        """

        is_meta_text: bool

    passage_checker = llm.with_structured_output(MetaTextCheckResponse)
    passage_check_instruction = [
        SystemMessage(
            content="""あなたはAIの研究者のために, 英語論文のMarkdownテキストを翻訳します.
論文全体を一度に翻訳することは難しいため, 英語論文を段落ごとに翻訳します.
ただし, 論文の本文以外のメタテキストは翻訳する必要はないので, 段落を翻訳する前に, 与えたテキストが論文のメタテキストかどうかを判断します.
論文のメタテキストとは, 著者名, 数式, 図のリンクや, テーブル, モデルの出力例, 数式, 参考文献です.
次に与えた英語論文のMarkdown形式のパッセージが, 論文のメタテキストかどうかを判断してください.

"""
        )
    ]
    for idx, passage in tqdm(
        enumerate(state.passages), total=len(state.passages), desc="Detect meta text"
    ):
        passage_check = passage_checker.invoke(
            passage_check_instruction + [HumanMessage(content=passage.original)],
            config={"callbacks": state.callbacks},
        )
        state.passages[idx].is_meta_text = passage_check.is_meta_text

    return state


def extract_terms(state: TranslatorState) -> TranslatorState:
    messages = [
        SystemMessage(
            content="あなたはAIの研究者です. 次の英語の論文に含まれる専門用語を抜き出したあと, 専門用語を日本語に翻訳してください. 翻訳文以外は出力しないでください."
        ),
    ]

    class TermExtractorResponse(BaseModel):
        terms: list[TechnicalTerm]

    term_extractor = llm.with_structured_output(TermExtractorResponse)

    terms = {}
    for passage in tqdm(state.passages, desc="Extract terms"):
        if passage.is_meta_text:
            continue
        ret = term_extractor.invoke(
            messages + [HumanMessage(content=passage.original)],
            config={"callbacks": state.callbacks},
        )
        for term in ret.terms:
            terms[term.original] = term
    state.terms = []
    for key in sorted(list(terms.keys())):
        state.terms.append(terms[key])

    return state


def refiner(passage: str, callbacks: list) -> str:
    prompts = [
        SystemMessage(
            content="""あな他はAIの研究者のために論文を校正します.
次に示す日本語の校正ガイドに従って, 与えられた論文のMarkdown形式の文章を校正してください.
出力には校正結果のテキスト以外の一切の文字列を返さないでください.

---

## 日本語校正ガイド

- カッコの前後は半角スペース
  - 正しい例: " (例) ".
  - 誤った例: "(例)", "（例）".
- 句読点は点丸"、。"ではなく, ピリオド".", カンマスペース ", "を使う.
  - 正しい例: 文章は, ピリオドで終わる.
  - 誤った例: 文章は、ピリオドで終わる。
- 文末に括弧がつく場合, 閉じ括弧の後に句点を打つ.
  - 正しい例: あいつはイケメンだ (自称).
  - 誤った例: あいつはイケメンだ. (自称)
- 見出しにはピリオドをつけず, 可能であれば体言止めにする.
- リストについて, 文であればピリオド"."で終わらせる. 
  - 例外: 体言であればピリオドはなくても良いが, リストの塊ごとにそろえる.
- ですます調は, 原則用いず, だである調を使う.
- 一文はできるだけ50字以内に収める.
- 主語と述語を近づける.
- 主語と述語を対応させる.


### 文章の改行について

段落の文章について, 1つの文は1行に書き, 文ごとに改行する. 
1文ごとに必ず改行する.
箇条書きや, 図表の文や, 画像のキャプションなどは改行は元の文章のままで良い.

#### 悪い例

この研究では, 新しい機械学習フレームワークを提案する. 提案する手法は, 特にスケーラビリティと実行効率で優れている.

#### 良い例

この研究では, 新しい機械学習フレームワークを提案する.
提案する手法は, 特にスケーラビリティと実行効率で優れている.


---
"""
        ),
        HumanMessage(content=passage),
    ]
    ret = llm.invoke(prompts, config={"callbacks": callbacks})
    return ret.content


def generate_translation_instruction(passage: str, terms: list[TechnicalTerm]) -> str:
    existed_terms = []
    for term in terms:
        if passage.find(term.original) != -1:
            existed_terms.append(f"- {term.original}")
    base_prompt = """あなたはAIの研究者です. 
次の英語論文のMarkdownテキストを日本語に翻訳してください.
翻訳文は, 翻訳スタイルガイドに従ってください.
翻訳結果以外のテキストは何も出力しないでください.

---

## 翻訳スタイルガイド

- 括弧の前後は半角スペース: " (例) ".
- 句読点は"。、"ではなく, ピリオド".", カンマスペース ", "を使う.
- 文末に括弧がつく場合, 閉じ括弧の後にピリオドを打つ.
  - 例: あいつはイケメンだ (自称).
- 見出しにはピリオドをつけず, 可能であれば体言止めにする.
- リストについて, 文であればピリオド"."で終わらせる. 
  - 例外: 体言であればピリオドはなくても良いが, リストの塊ごとにそろえる.
- ですます調は, 会話文以外の通常分では原則用いず, だである調を使う.
- 技術用語の長音は省く.
  - 技術用語なので省く例: コンピュータ.
  - 技術用語ではないので省かない例: シャッター.
- 一文はできるだけ50字以内に収める.
- 人名は日本語に翻訳せず, 原文のまま書く.

---
"""

    if len(existed_terms) > 0:
        existed_terms = "\n".join(existed_terms)
        base_prompt = f"""{base_prompt}
- 専門用語リストにある語彙は翻訳せずに原文のまま書く.

### 専門用語リスト

{existed_terms}

---
"""
    else:
        base_prompt = f"{base_prompt}\n\n---\n"
    return base_prompt


def translate_passage(state: TranslatorState) -> TranslatorState:
    for idx, passage in tqdm(
        enumerate(state.passages), total=len(state.passages), desc="Translate"
    ):
        if passage.is_meta_text:
            state.passages[idx].translated = passage.original
        else:
            translation_instruction = [
                SystemMessage(
                    content=generate_translation_instruction(
                        passage=passage.original, terms=state.terms
                    )
                ),
                HumanMessage(content=passage.original),
            ]
            ret = llm.invoke(
                translation_instruction, config={"callbacks": state.callbacks}
            )
            ret = refiner(ret.content, callbacks=state.callbacks)

            state.passages[idx].translated = ret
    return state


translator_graph_builder = StateGraph(TranslatorState)
translator_graph_builder.add_node("text_parser", parse_text)
translator_graph_builder.add_edge(START, "text_parser")
translator_graph_builder.add_node("meta_text_checker", detect_meta_text)
translator_graph_builder.add_edge("text_parser", "meta_text_checker")
translator_graph_builder.add_node("term_extractor", extract_terms)
translator_graph_builder.add_edge("meta_text_checker", "term_extractor")
translator_graph_builder.add_node("translator", translate_passage)
translator_graph_builder.add_edge("term_extractor", "translator")
translator_graph_builder.add_edge("translator", END)
translator = translator_graph_builder.compile()

app = Typer()


@app.command()
def translate(
    paper_path: str = "paper.md",
):
    """
    Translate the given paper.
    """
    paper_path = Path(paper_path)
    paper = paper_path.read_text()
    callback = UsageMetadataCallbackHandler()
    ret = translator.invoke({"original_text": paper, "callbacks": [callback]})
    ret = TranslatorState(**ret)
    print(callback.usage_metadata)
    paper_ja = "\n\n".join([passage.translated for passage in ret.passages])
    paper_ja_path = paper_path.parent / f"{paper_path.name.replace('.md', '_ja.md')}"
    paper_ja_path.write_text(paper_ja)


if __name__ == "__main__":
    app()
