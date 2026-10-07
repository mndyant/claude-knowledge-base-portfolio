const $ = (selector) => document.querySelector(selector);
const state = { loading: false, sourceNotes: false, retrieval: "unknown", ready: null };
const queryInput = $("#query");
const searchButton = $("#search-button");
const results = $("#results");
const message = $("#message");

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[char]));
}

function sourceUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

function showMessage(title, detail = "") {
  message.hidden = false;
  message.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(detail)}</p>`;
}

function setLoading(loading) {
  state.loading = loading;
  searchButton.disabled = loading;
  searchButton.textContent = loading ? "検索中…" : "検索";
  document.querySelectorAll("[data-query]").forEach((button) => { button.disabled = loading; });
  $("#results-section").setAttribute("aria-busy", String(loading));
}

function renderResults(payload, query) {
  $("#results-heading").hidden = false;
  $("#results-title").textContent = `「${query}」の検索結果`;
  $("#search-meta").textContent = `${payload.results.length}件 · ${Number(payload.elapsed_ms).toLocaleString()} ms`;
  if (!payload.results.length) {
    showMessage("一致する資料がありません", state.sourceNotes
      ? "対象はストリーミング・画像入力・ツール呼び出しです。日本語は2文字以上の語で試してください。"
      : "キーワードやカテゴリを変えて試してください。");
    return;
  }
  message.hidden = true;
  $("#announcer").textContent = `「${query}」に関連する資料が${payload.results.length}件見つかりました。`;
  const retrieval = payload.retrieval || state.retrieval;
  const sourceNotes = state.sourceNotes || payload.results.every((item) => item.source?.startsWith("demo-notes/"));
  results.innerHTML = payload.results.map((item, index) => {
    const title = item.japanese_title || item.heading_path || item.section || item.source?.split(/[\\/]/).pop() || "資料";
    const summary = item.japanese_summary || item.summary_short || "";
    const url = sourceUrl(item.source_url);
    const score = Number(item.score);
    const scoreText = Number.isFinite(score)
      ? `${retrieval === "bm25" ? "BM25" : "類似度"} ${score.toFixed(retrieval === "bm25" ? 2 : 3)}` : "";
    return `<article class="result-card" id="source-${index + 1}">
      <div class="result-top"><h3>${escapeHtml(title)}</h3><span class="result-number">${String(index + 1).padStart(2, "0")}</span></div>
      ${summary ? `<p class="summary">${escapeHtml(summary)}</p>` : ""}
      <details class="original"><summary>${sourceNotes ? "英語の独自要約を読む" : "本文を読む"}</summary>
        <div class="result-content">${escapeHtml(item.content)}</div>
        <p class="source-path">${escapeHtml(item.source)}</p>
      </details>
      <div class="result-bottom">${url ? `<a class="source-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">公式出典を読む ↗</a>` : ""}<span class="score">${scoreText}</span></div>
    </article>`;
  }).join("");
  if (payload.answer) {
    $("#answer").hidden = false;
    const answer = escapeHtml(payload.answer).replace(/\[(\d+)\]/g, '<a href="#source-$1">[$1]</a>').replace(/\n/g, "<br>");
    $("#answer").innerHTML = `<h3>出典付き回答</h3><div>${answer}</div><p>生成された回答です。引用先の資料も確認してください。</p>`;
  }
}

async function runSearch(query) {
  const cleanQuery = query.trim();
  if (!cleanQuery || state.loading) return;
  if (cleanQuery.length > 4000) { showMessage("質問が長すぎます", "4000文字以内で入力してください。"); return; }
  queryInput.value = cleanQuery;
  setLoading(true);
  results.innerHTML = "";
  $("#announcer").textContent = "";
  $("#answer").hidden = true;
  $("#results-heading").hidden = true;
  showMessage("関連する資料を探しています…");
  try {
    await state.ready;
    const response = await fetch("/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: cleanQuery,
        top_k: state.sourceNotes ? 3 : Number($("#top-k").value),
        category: state.sourceNotes ? null : ($("#category").value || null),
        generate_answer: !state.sourceNotes && $("#generate-answer").checked }),
      signal: AbortSignal.timeout(60000)
    });
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error(response.ok ? "検索結果を読み取れませんでした。再度お試しください。" : `検索に失敗しました（HTTP ${response.status}）`); }
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `検索に失敗しました（HTTP ${response.status}）`);
    renderResults(payload, cleanQuery);
  } catch (error) {
    showMessage("検索できませんでした", error.name === "TimeoutError"
      ? "接続がタイムアウトしました。少し待ってから再度検索してください。"
      : error instanceof TypeError ? "サーバーへ接続できません。通信状態を確認して再度お試しください。" : error.message);
  } finally { setLoading(false); }
}

async function loadStatus() {
  try {
    const response = await fetch("/health", { signal: AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error("Index unavailable");
    const health = await response.json();
    state.sourceNotes = health.demo_mode === "source-notes";
    state.retrieval = health.retrieval || "e5";
    $("#index-status").textContent = `${Number(health.total_documents).toLocaleString()}件の検索対象 · ${state.retrieval === "bm25" ? "BM25" : "e5意味検索"}`;
    if (state.sourceNotes) {
      const publicDemo = state.retrieval === "bm25";
      $("#mode-label").textContent = publicDemo ? "公開デモ · キーワード検索" : "ローカルデモ · 意味検索";
      $("#intro").textContent = "ストリーミング・画像入力・ツール呼び出しを、公式出典付きの要約から検索。";
      const sourceLabel = Number.isInteger(health.source_pages) ? `公式${health.source_pages}ページ` : "公式資料";
      $("#scope-note").textContent = `${sourceLabel}を基にした独自要約${health.total_documents}件。回答生成なし。最新モデル・料金は対象外。${health.sources_checked_at ? `出典確認：${health.sources_checked_at}。` : ""}`;
      $("#generate-answer").disabled = true;
    } else {
      $("#advanced-options").hidden = false;
      $("#scope-note").textContent = "登録済みの資料を検索します。回答生成は検索オプションから選べます。";
    }
  } catch {
    $("#index-status").textContent = "検索サーバーへの接続を確認できません";
    $("#scope-note").textContent = "接続状態を確認できませんでした。サーバーの起動・通信状態を確認してください。";
  }
}

$("#search-form").addEventListener("submit", (event) => { event.preventDefault(); runSearch(queryInput.value); });
document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => runSearch(button.dataset.query));
});
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !document.activeElement.matches("input, textarea, select, [contenteditable]")) {
    event.preventDefault(); queryInput.focus();
  }
});
state.ready = loadStatus();
const initialQuery = new URLSearchParams(location.search).get("q");
if (initialQuery) runSearch(initialQuery);
