"""公式資料に基づくデモ用の独自要約。原文の転載・架空のAPI仕様ではない。"""

CHECKED_AT = "2026-10-07"
BASE = "https://platform.claude.com/docs/en/"

# English text and Japanese summaries are written for this project, not quotations.
NOTES = [
    {
        "id": "streaming-request",
        "title": "Streaming messages / Request",
        "url": BASE + "build-with-claude/streaming",
        "summary": "応答を少しずつ受け取るには、Messages APIでstreamをtrueにします。SSEで届くイベントを処理でき、Python SDKではtext_streamからテキストを読み取れます。",
        "content": "To receive a Claude Messages API response incrementally, enable the stream option. The transport is server-sent events (SSE). In the Python SDK, use client.messages.stream and iterate over text_stream to consume text as it arrives. Streaming lets an application display partial output before the message has finished.",
    },
    {
        "id": "streaming-events",
        "title": "Streaming messages / Event lifecycle",
        "url": BASE + "build-with-claude/streaming",
        "summary": "message_startで始まり、content_blockの開始・差分・終了、message_deltaを経てmessage_stopで完了します。pingやerror、新しいイベント型への対応も必要です。",
        "content": "A streamed response begins with message_start. Content blocks have start, delta and stop events. Message-level updates arrive as message_delta, followed by message_stop. A stream can also contain ping and error events. Applications consuming raw SSE should handle unexpected event types gracefully instead of assuming every event contains text.",
    },
    {
        "id": "vision-input",
        "title": "Vision / Send images to Claude",
        "url": BASE + "build-with-claude/vision",
        "summary": "画像はMessages APIのimageブロックで渡します。Claude APIではbase64データ、画像URL、Files APIで取得したfile_idを使えます。画像をテキストの前に置く構成が推奨されています。",
        "content": "Images are supplied in image content blocks within user messages. On the Claude API, an image source may be base64 data, an online image URL, or a file_id from the Files API. Base64 sources include media_type and data. For prompts containing both an image and a question, prefer placing the image before the text.",
    },
    {
        "id": "vision-formats",
        "title": "Vision / Formats and repeated images",
        "url": BASE + "build-with-claude/vision",
        "summary": "JPEG・PNG・GIF・WebPを扱えます。アニメーションは先頭フレームのみです。繰り返し使う画像はFiles APIへ一度アップロードし、file_idを参照するとリクエストを小さくできます。",
        "content": "Supported image formats include JPEG, PNG, GIF and WebP. Animated images are processed using their first frame. When an image is reused, upload it to the Files API and refer to its file_id in later requests. This avoids repeatedly placing the base64 image bytes in the conversation payload.",
    },
    {
        "id": "tools-roundtrip",
        "title": "Tool use / Client tool round trip",
        "url": BASE + "agents-and-tools/tool-use/overview",
        "summary": "クライアントツールでは、Claudeがtool_useブロックで呼び出しを要求します。アプリが処理を実行し、対応するtool_use_idを付けたtool_resultを返すと、Claudeが結果を使って回答できます。",
        "content": "For client tools, Claude returns a tool_use content block describing the requested operation. Your application executes the tool, then sends a tool_result in a user message. The result references the original call using tool_use_id. Include the assistant tool call in the conversation before returning the result so Claude can continue with the outcome.",
    },
    {
        "id": "tools-execution",
        "title": "Tool use / Client and server tools",
        "url": BASE + "agents-and-tools/tool-use/overview",
        "summary": "クライアントツールの実行場所は自分のアプリです。Anthropicがスキーマを定義するクライアントツールも同様です。サーバーツールはAnthropic側で実行されます。",
        "content": "Client tools run in your application, including custom tools and tools whose schemas are provided by Anthropic. Your code handles execution and returns the results. Server tools execute on Anthropic infrastructure. Choosing a tool therefore also determines where its operation is performed and whether the application needs its own execution handler.",
    },
]


def source_for(note: dict[str, str]) -> str:
    """独自要約と通常の取得原文を区別できるソース名を返す。"""
    return f"demo-notes/docs/{note['id']}.md"
