# Q&A実例集：このRAGシステムは実際どう答えるか

`docs/EVAL.md` では Recall@k・MRR・groundedness率・citation precision といった
数値でこのシステムの検索・回答品質を評価しています。ただし数値表だけでは
「実際どんな質問に、どんな回答と根拠を返すのか」が伝わりにくいため、本ドキュメントでは
評価用データセット（15問）の中から代表例を選び、質問・生成された回答・引用元・
判定結果をそのまま会話ログのように並べています。

回答はすべて `eval/cache/answers.jsonl`（生成済みキャッシュ）、判定結果はすべて
`eval/cache/judgements.jsonl`（LLM-judgeによるclaim単位の検証結果）からの引用で、
加筆・修正はしていません。うまくいった例だけでなく、検索が弱かった例も正直に載せています。

集計値の全体像（groundedness率 1.0、citation precision 1.0、trap abstention accuracy
0.6667 など、15問・69claim分の内訳）は [docs/EVAL.md](EVAL.md) の
「回答生成とhallucination評価」セクションを参照してください。

---

## 例1: 画像をAPIに渡す方法（易問・日本語）

**質問**: 「画像をAPIに渡してClaudeに見せる方法」

**回答**:
> Claudeに画像を渡して分析させる方法は、主に以下の3つのアプローチがあります\[2\]。
>
> 1. **base64エンコードされた画像として渡す**
>    `image`コンテンツブロック内にbase64エンコードしたデータを含める方法です\[2\]。なお、Amazon BedrockやVertex AIを使用している場合は、この方法のみが利用可能です\[2\]。
>
> 2. **オンライン上の画像URLを参照する**
>    画像がホストされているURLを直接指定する方法です\[2\]\[4\]。APIリクエストにおいて、ソースタイプを「url」として指定します\[4\]。
>
> 3. **Files APIを利用する**
>    ファイルを一度アップロードし、そのファイルを複数回利用する方法です\[2\]。
>
> なお、APIを通じて画像を送信する際は、標準的なマルチターン会話構造の一部として、`user`ロールのメッセージ配列の中に新しい画像を追加します\[5\]。また、最適な結果を得るために、JPEG、PNG、GIF、WebPといったサポートされているフォーマットを使用し、画像の鮮明さを確保することなどが推奨されています\[1\]\[4\]。

**引用元**:

- \[1\] <https://platform.claude.com/docs/en/build-with-claude/vision> — 「Vision > Before you upload > Ensure image quality」
- \[2\] <https://platform.claude.com/docs/en/build-with-claude/vision> — 「Vision > Prompt examples > About the prompt examples」
- \[4\] <https://platform.claude.com/docs/en/build-with-claude/vision> — 「Vision > FAQ」
- \[5\] <https://platform.claude.com/docs/en/build-with-claude/vision> — 「Vision > Prompt examples > Files API image example」

**評価**: ✅ groundedness確認済み（claim 8件すべてsupported）/ citation precision 100%。典型的な易問（トップレベルのVisionページで完結）が、期待どおりクリーンに解けた例。

---

## 例2: 組織のレートリミット確認（難問・日本語・複数ページ）

**質問**: 「組織のAPIレートリミットの上限を確認する方法」

**回答**:
> 組織のAPIレートリミットの上限を確認するには、以下のエンドポイントに対してGETリクエストを送ります\[4\]。
>
> `GET /v1/organizations/rate_limits`
>
> このエンドポイントには、以下のクエリパラメータを指定してフィルタリングすることが可能です\[4\]。
>
> - `group_type`: グループタイプでフィルタリングします（`"model_group"`、`"batch"`、`"token_count"`、`"files"`、`"skills"`、`"web_search"`のいずれか）\[4\]。
> - `model`: 特定のモデルに関連するエントリのみを取得します（フルモデル名またはエイリアスが使用可能）\[4\]\[6\]。モデルが見つからない場合やレートリミットが設定されていない場合は404エラーが返されます\[4\]\[6\]。
> - `page`: 前回のレスポンスから得られた `next_page` カーソルを使用してページネーションを行います\[4\]。
>
> リクエストの例は以下の通りです\[1\]\[2\]。
>
> ```http
> curl https://api.anthropic.com/v1/organizations/rate_limits \
>     -H 'anthropic-version: 2023-06-01' \
>     -H "X-Api-Key: $ANTHROPIC_ADMIN_API_KEY"
> ```

**引用元**:

- \[1\] <https://platform.claude.com/docs/en/api/admin/rate_limits/list> — 「List Organization Rate Limits > List Organization Rate Limits > Example」
- \[2\] <https://platform.claude.com/docs/en/api/admin> — 「Rate Limits > List Organization Rate Limits > Example」
- \[4\] <https://platform.claude.com/docs/en/api/admin> — 「Rate Limits > List Organization Rate Limits」
- \[6\] <https://platform.claude.com/docs/en/manage-claude/rate-limits-api> — 「Rate Limits API > Organization rate limits > Look up the limits for a specific model」

**評価**: ✅ groundedness確認済み（claim 6件すべてsupported）/ citation precision 100%。この質問はAPIリファレンスページ（`/api/admin`）とガイドページ（`/manage-claude/rate-limits-api`）の**両方**に正解チャンクが存在する設計になっており、実際に両方から正しく引けている（データセットのnotesにも「組織向け/ワークスペース向けの2ページが両方とも正解」と明記）。

---

## 例3: ツール入力のストリーミング受信（難問・英語）

**質問**: "how to receive partial tool input as it streams in"

**回答**:
> Fine-grained tool streamingを利用して、ツール入力がストリーミングされる際に部分的な入力を受け取る方法は以下の通りです。
>
> `tool_use` コンテンツブロックがストリーミングされる際、最初に出力される `content_block_start` イベントにはプレースホルダーとして `input: {}` が含まれます。実際の入力値は、一連の `input_json_delta` イベントを通じて、`partial_json` 文字列フラグメントとして断続的に到着します\[2\]。
>
> これらのフラグメントを組み立てるには、以下の手順に従います\[2\]。
>
> 1. **初期化**: `type: "tool_use"` を持つ `content_block_start` イベントを受け取ったら、空の文字列（例: `input_json = ""`）を初期化します\[2\]。
> 2. **蓄積**: `type: "input_json_delta"` を持つ各 `content_block_delta` イベントを受け取るたびに、`partial_json` を現在の文字列に追加していきます\[2\]。
> 3. **完了と解析**: `content_block_stop` イベントを受け取った時点で、蓄積した文字列をパースします\[2\]。
>
> SDKにアクセサリ（accumulator helper）が用意されている場合は、それを使用することが推奨されます\[2\]\[4\]。手動で処理を行う必要がある場合や、ブロックが終了する前に部分的な入力に対して反応（プログレスインジケーターの表示など）したい場合は、上記のパターンを使用してください\[2\]\[4\]。
>
> 注意点として、この機能ではサーバーサイドでのJSONバリデーションが行われないため、受信するJSONが不完全または無効である可能性があることに留意してください\[1\]\[5\]。

**引用元**:

- \[1\] <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming> — 「Fine-grained tool streaming」
- \[2\] <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming> — 「Fine-grained tool streaming > Accumulating tool input deltas」
- \[4\] <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming> — 「Fine-grained tool streaming」（Accumulating tool input deltasの続き）
- \[5\] <https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming> — 「Fine-grained tool streaming > How to use fine-grained tool streaming」

**評価**: ✅ groundedness確認済み（claim 6件すべてsupported）/ citation precision 100%。質問文には「fine-grained tool streaming」という機能名が一切出てこない（"partial tool input as it streams in" という説明的な言い回しのみ）にもかかわらず、正しいページを引き当てている難問。

---

## 例4: 存在しない`retry_policy`パラメータ（トラップ・日本語）

**質問**: 「Claude APIのmessagesリクエストにある'retry_policy'パラメータで自動リトライ回数を設定する方法を教えてください」

**回答**:
> ご質問いただいた「messagesリクエストにおける『retry_policy』パラメータ」および「自動リトライ回数の設定方法」については、提供されたドキュメントからは確認できません。
>
> コンテキスト内には、Claude APIにおけるリトライ設定や`retry_policy`パラメータに関する言及は存在しません。

**引用元**: なし（引用すべき根拠が無いため、citationを一切付けずに拒否している）

**評価**: ✅ 正しく回答を拒否（abstain）。`retry_policy` はAnthropic APIに実在しないパラメータ。なぜこれが重要かというと、もっともらしい嘘のパラメータ名を投げられたときに「それらしい回答をでっち上げてしまう」のがRAGシステムの典型的な失敗パターンだからで、この例はそれを避けられている。トラップ3問（t01/t02/t03）のうち、citationを一切付けずに完全に拒否したのはこのt01のみ。

---

## 例5: 存在しない`creativity_level`パラメータ（トラップ・英語）

**質問**: "how do I set the 'creativity_level' parameter to control how imaginative Claude's responses are"

**回答**:
> 提供されたドキュメントには、「creativity_level」という名前のパラメータは存在しません。
>
> Claudeの回答のランダム性（創造性）を制御するには、`temperature` パラメータを使用します\[1\]\[2\]\[3\]\[4\]\[5\]\[6\]。
> このパラメータの設定に関する詳細は以下の通りです。
>
> - **デフォルト値**: `1.0` \[1\]\[2\]\[3\]\[4\]\[5\]\[6\]
> - **範囲**: `0.0` から `1.0` \[1\]\[2\]\[3\]\[4\]\[5\]\[6\]
> - **使い方**: 分析的あるいは多肢選択式のタスクには `0.0` に近い値を、創造的・生成的なタスクには `1.0` に近い値を使用することが推奨されています\[1\]\[2\]\[3\]\[4\]\[5\]\[6\]。
> - **注意点**: `temperature` を `0.0` に設定しても、結果が完全に決定論的（再現性が保証されたもの）になるわけではありません\[1\]\[2\]\[3\]\[4\]\[5\]\[6\]。

**引用元**（いずれも各言語SDK/APIリファレンスの`temperature`パラメータ説明）:

- \[1\] <https://platform.claude.com/docs/en/api/ruby/messages/batches> — 「Batches > Create a Message Batch > Parameters」
- \[2\] <https://platform.claude.com/docs/en/api/cli/messages/create> — 「Create a Message (cli) > Create a Message > Parameters」
- \[3\] <https://platform.claude.com/docs/en/api/ruby/messages> — 「Messages > Create a Message > Parameters」
- \[4\] <https://platform.claude.com/docs/en/api/cli/messages> — 「Messages > Create a Message > Parameters」
- \[5\] <https://platform.claude.com/docs/en/api/php/messages/create> — 「Create a Message (php) > Create a Message > Parameters」
- \[6\] <https://platform.claude.com/docs/en/api/python/beta> — 「Messages > Create a Message > Parameters」

**評価**: ✅ hallucinationなし（claim 5件すべてsupported・citation精度100%）。架空の `creativity_level` パラメータの存在は明確に否定しつつ、実在する近縁パラメータ（`temperature`）へ正しく誘導している。EVAL.mdの評価スクリプト上は「abstained: false」（回答内容に踏み込んだため）と判定されるが、内容を読むとhallucinationはしておらず、「架空機能を否定＋実在機能へ誘導」という単純な二値のabstention指標では拾いきれない、実務上は最も望ましい振る舞いの一例。

---

## 例6: Claude自身での埋め込みベクトル生成（検索の限界・正直な失敗例・日本語）

**質問**: 「Claude自体で文章の埋め込みベクトルを作ることはできますか」

**回答**:
> ご質問いただいた内容について、提供されたドキュメントを確認しましたが、Claude自体で文章の埋め込みベクトル（embedding）を作成する機能については記載がありません。
>
> ドキュメントには、メッセージ作成時のパラメータ（`temperature` や `thinking` 設定など）や、トークン数のカウント方法については説明されていますが\[1\]\[2\]\[3\]\[4\]\[5\]\[6\]、埋め込みに関する言及は確認できませんでした。

**引用元**（付いてはいるが、いずれも質問への回答にはなっていない）:

- \[1\] <https://platform.claude.com/docs/en/api/python/beta> — 「Messages > Create a Message > Parameters」
- \[2\] <https://platform.claude.com/docs/en/api/python/beta/messages/batches> — 「Batches > Create a Message Batch > Parameters」
- \[3\] <https://platform.claude.com/docs/en/api/java/beta/messages/batches/create> — 「Create a Message Batch (Beta) (Java) > Create a Message Batch > Parameters」

**評価**: ⚠️ **不必要な回答拒否**（この質問は`eval/qa_dataset.jsonl`上は `answerable: true`）。
この質問の本来の正解は「Anthropicは自社の埋め込みAPIを提供しておらず、Voyage AIの利用を案内している」ですが、検索段階でEmbeddings/Voyage AIページのチャンクがtop-6コンテキストに一件も含まれておらず、生成モデルは「見つからない」と正直に答える他なかった。
`docs/EVAL.md` の分析によれば、これはanswerable 12問中3件（q003・q017・q033）発生した「不必要な回答拒否」の一つで、**原因は生成側の過剰な保守性ではなく検索側の取りこぼし**（該当ページの記述が薄く他セクションと混ざっていた可能性）。
隠さずに書くと：hallucinateして「Voyage AIを使ってください」ともっともらしく答えてしまうより、分からないものを分からないと言う方向に倒れたのは安全側の失敗ではあるものの、ユーザー体験としては「本来答えられたはずの質問に答えられなかった」という損失であり、改善の優先課題として`docs/EVAL.md`にも明記されている。

---

## 例7: Claudeの画像認識の弱点（難問・日本語・サブセクション特定＋複数ページ）

**質問**: 「Claudeが画像を見て苦手なこと・間違えやすいことは何ですか」

**回答**:
> Claudeの画像認識や画像に関連するタスクにおいて、苦手なことや間違えやすいことは以下の通りです。
>
> - **人物の特定**: 画像内の人物に名前を付けることはできず、拒否されます\[2\]。
> - **精度の問題**: 低品質、回転している、または200ピクセル未満の非常に小さい画像では、幻覚を見たり間違いを犯したりする可能性があります\[2\]。
> - **空間的推論**: 精密な位置特定やレイアウトが必要なタスク（アナログ時計の読み取りやチェスの駒の正確な位置の記述など）は苦手としています\[2\]。
> - **数のカウント**: 画像内のオブジェクトの数を概算することはできますが、特に小さなオブジェクトが多数ある場合など、常に正確であるとは限りません\[2\]。
> - **AI生成画像の判定**: 画像がAIによって生成されたものかどうかを判断することはできず、質問されても誤った回答をする可能性があるため、AI生成画像の検出には頼らないでください\[2\]。
> - **不適切なコンテンツ**: ポリシーに違反する不適切または露骨な画像は処理されません\[2\]。
> - **医療画像**: 一般的な医療画像の分析は可能ですが、CTやMRIのような複雑な診断用スキャンの解釈には設計されておらず、専門的な医療診断の代わりにはなりません\[2\]。
> - **機密性や高い精度を要する作業**: 完璧な精度や機密性の高い分析が必要なタスクには、人間による監視なしでClaudeを使用しないでください\[2\]。
>
> また、コンピュータ使用ツール（Computer use tool）に関連するコンテキストでは、操作対象のクリック精度や座標指定において間違いや幻覚が発生する可能性があること\[5\]、Webページや画像内に含まれる指示がClaudeの本来の指示を上書きして誤作動を引き起こすリスク（プロンプトインジェクション等）があること\[5\]も挙げられています。

**引用元**:

- \[2\] <https://platform.claude.com/docs/en/build-with-claude/vision> — 「Vision > Limitations」
- \[5\] <https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool> — 「Computer use tool > Understand computer use limitations」

**評価**: ✅ groundedness確認済み（claim 10件すべてsupported）/ citation precision 100%。正解チャンクがVisionページ内の「Limitations」という特定サブセクションのみに存在する難問だが、それを的確に引き当てた上、関連する別ページ（Computer use toolの制限事項）まで自発的に補足しており、単一ページの表層検索では出せない厚みのある回答になっている。

---

## まとめ

以上7例は、評価対象15問（易問・難問・トリック質問・トラップ質問・日英混在）から
成功例と失敗例の両方を選んで抜粋したものです。全15問の質問一覧・claimごとの
判定内訳・集計値（groundedness率1.0、citation precision 1.0、trap abstention
accuracy 0.6667など）は [docs/EVAL.md](EVAL.md) を参照してください。
