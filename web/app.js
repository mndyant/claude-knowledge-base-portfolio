const state={category:"",loading:false};
const $=(selector)=>document.querySelector(selector);
const queryInput=$("#query"),searchForm=$("#search-form"),searchButton=$("#search-button");
const suggestions=$("#suggestions"),results=$("#results"),stateCard=$("#state-card"),answerPanel=$("#answer-panel");
const resultsTitle=$("#results-title"),sectionKicker=$("#section-kicker"),searchMeta=$("#search-meta");

function escapeHtml(value=""){return String(value).replace(/[&<>'"]/g,(char)=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));}
function categoryFromSource(source){return source.match(/knowledge[\\/]([^\\/]+)/)?.[1]||"source";}
function titleFor(item){return item.japanese_title||item.heading_path||item.section||item.summary_short||item.source.split(/[\\/]/).pop()?.replace(/\.md$/i,"")||"Untitled source";}
function setLoading(loading){
  state.loading=loading;searchButton.disabled=loading;
  if(!loading)return;
  suggestions.hidden=true;results.hidden=true;answerPanel.hidden=true;stateCard.hidden=false;
  stateCard.innerHTML='<div class="spinner"></div><strong>Searching the knowledge base</strong><p>\u95a2\u9023\u3059\u308b\u4e00\u6b21\u60c5\u5831\u3092\u63a2\u3057\u3066\u3044\u307e\u3059...</p>';
}
function showMessage(title,message){
  suggestions.hidden=true;results.hidden=true;answerPanel.hidden=true;stateCard.hidden=false;
  stateCard.innerHTML=`<strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p>`;
}
function renderAnswer(answer){
  if(!answer){answerPanel.hidden=true;answerPanel.innerHTML="";return;}
  const linked=escapeHtml(answer).replace(/\[(\d+)\]/g,'<a href="#source-$1">[$1]</a>').replace(/\n/g,"<br>");
  answerPanel.hidden=false;
  answerPanel.innerHTML=`<div class="answer-label">RAG AGENT ANSWER</div><h3>公式情報に基づく回答</h3><div class="answer-body">${linked}</div><p class="answer-note">回答中の番号から、根拠となる公式資料を確認できます。</p>`;
}
function renderResults(payload,query){
  stateCard.hidden=true;suggestions.hidden=true;results.hidden=false;
  sectionKicker.textContent="SEARCH RESULTS";resultsTitle.textContent=`"${query}"`;
  searchMeta.textContent=`${payload.total} results - ${payload.elapsed_ms.toLocaleString()} ms`;
  if(!payload.results.length){showMessage("No results found","\u30ad\u30fc\u30ef\u30fc\u30c9\u3084\u30ab\u30c6\u30b4\u30ea\u3092\u5909\u3048\u3066\u3001\u3082\u3046\u4e00\u5ea6\u304a\u8a66\u3057\u304f\u3060\u3055\u3044\u3002");return;}
  renderAnswer(payload.answer);
  results.innerHTML=payload.results.map((item,index)=>{
    const score=Math.max(0,Math.min(1,Number(item.score)||0));
    const summary=item.japanese_summary?`<p class="summary">${escapeHtml(item.japanese_summary)}</p>`:(item.summary_short&&item.summary_short!==titleFor(item)?`<p class="summary">${escapeHtml(item.summary_short)}</p>`:"");
    return `<article class="result-card" id="source-${index+1}" style="animation-delay:${index*45}ms">
      <div class="result-top"><span class="category-badge">${escapeHtml(categoryFromSource(item.source))}</span>
      <span class="citation-badge">引用 ${index+1}</span>
      ${item.has_code?'<span class="category-badge">code</span>':""}
      <span class="score"><span class="score-bar"><i style="width:${score*100}%"></i></span>${Math.round(score*100)}% match</span></div>
      <h3>${escapeHtml(titleFor(item))}</h3>${summary}
      <details class="result-original"><summary>英語の原文を表示</summary><div class="result-content">${escapeHtml(item.content)}</div></details>
      <div class="result-actions"><span class="source-path" title="${escapeHtml(item.source)}">${escapeHtml(item.source)}</span>
      ${item.source_url?`<a class="source-link" href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer">公式ページで詳しく読む ↗</a>`:""}
      <button class="text-button expand-button" type="button">詳しく見る</button>
      <button class="text-button copy-button" type="button">コピー</button></div>
    </article>`;
  }).join("");
}
async function runSearch(query){
  const cleanQuery=query.trim();if(!cleanQuery||state.loading)return;queryInput.value=cleanQuery;setLoading(true);
  try{
    const response=await fetch("/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:cleanQuery,top_k:Number($("#top-k").value),category:state.category||null,generate_answer:$("#generate-answer")?.checked||false})});
    const payload=await response.json();if(!response.ok)throw new Error(payload.detail||"Search failed");
    renderResults(payload,cleanQuery);$("#results-section").scrollIntoView({behavior:"smooth",block:"start"});
  }catch(error){
    sectionKicker.textContent="CONNECTION ERROR";resultsTitle.textContent="Search unavailable";searchMeta.textContent="";
    showMessage("\u30ca\u30ec\u30c3\u30b8\u30d9\u30fc\u30b9\u306b\u63a5\u7d9a\u3067\u304d\u307e\u305b\u3093",error.message);
  }finally{setLoading(false);}
}
searchForm.addEventListener("submit",(event)=>{event.preventDefault();runSearch(queryInput.value);});
document.querySelectorAll("[data-query]").forEach((button)=>button.addEventListener("click",()=>runSearch(button.dataset.query)));
document.querySelectorAll(".filter-chip").forEach((button)=>button.addEventListener("click",()=>{
  document.querySelectorAll(".filter-chip").forEach((chip)=>chip.classList.remove("active"));
  button.classList.add("active");state.category=button.dataset.category;if(queryInput.value.trim())runSearch(queryInput.value);
}));
results.addEventListener("click",async(event)=>{
  const card=event.target.closest(".result-card");if(!card)return;
  if(event.target.matches(".expand-button")){const expanded=card.classList.toggle("expanded");const original=card.querySelector(".result-original");if(original)original.open=expanded;event.target.textContent=expanded?"閉じる":"詳しく見る";}
  if(event.target.matches(".copy-button")){
    await navigator.clipboard.writeText(card.querySelector(".summary")?.textContent||card.querySelector(".result-content").textContent);event.target.textContent="コピーしました";
    setTimeout(()=>{event.target.textContent="コピー";},1300);
  }
});
document.addEventListener("keydown",(event)=>{
  if(event.key==="/"&&!["INPUT","TEXTAREA"].includes(document.activeElement.tagName)){event.preventDefault();queryInput.focus();}
  if(event.key==="Escape")queryInput.blur();
});
document.querySelectorAll("[data-focus-search]").forEach((button)=>button.addEventListener("click",()=>queryInput.focus()));
const sidebar=$("#sidebar"),overlay=$("#overlay");
function toggleMenu(open){sidebar.classList.toggle("open",open);overlay.hidden=!open;$("#menu-button").setAttribute("aria-expanded",String(open));}
$("#menu-button").addEventListener("click",()=>toggleMenu(!sidebar.classList.contains("open")));overlay.addEventListener("click",()=>toggleMenu(false));
async function loadStatus(){
  const [healthResult,sourcesResult]=await Promise.allSettled([
    fetch("/health").then((response)=>response.ok?response.json():Promise.reject()),
    fetch("/sources").then((response)=>response.ok?response.json():Promise.reject())
  ]);
  if(healthResult.status==="fulfilled"){
    $("#health-dot").classList.add("online");$("#health-label").textContent="Index online";
    $("#document-count").textContent=healthResult.value.total_documents.toLocaleString();
  }else{$("#health-label").textContent="Index offline";}
  if(sourcesResult.status==="fulfilled"){
    $("#source-count").textContent=`${sourcesResult.value.total.toLocaleString()} sources`;
    if(sourcesResult.value.sources.length && sourcesResult.value.sources.every((source)=>source.startsWith("sample/"))){
      $(".eyebrow").textContent="LOCAL RAG · SAMPLE DATA";
      $(".hero h1").textContent="小さなデータで、検索の流れを確認。";
      $(".hero > p").textContent="自作のサンプル文書4件を実際のe5モデルで検索します。公式文書や個人用DBは使用していません。";
      $("#health-label").textContent="Sample index online";
      $("#generate-answer").disabled=true;
      $("#generate-answer").parentElement.textContent="サンプルデモ：外部LLMを使わない検索";
      const prompts=[
        ["トークン上限", "なぜチャンクのトークン数を測る必要がある？"],
        ["再開可能な処理", "中断したインデックス作成をどう再開する？"],
        ["安全なテスト", "本番のデータを変更せずにテストするには？"]
      ];
      document.querySelectorAll(".suggestion-card").forEach((button,index)=>{
        button.dataset.query=prompts[index][1];
        button.querySelector("strong").textContent=prompts[index][0];
        button.querySelector("small").textContent=prompts[index][1];
      });
    }
  }
}
loadStatus();
