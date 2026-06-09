/* Trợ lý Pháp luật Ma túy — frontend logic (streaming NDJSON + memory) */

const chat = document.getElementById("chat");
const hero = document.getElementById("hero");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const rerank = document.getElementById("rerank");

const messages = []; // memory: [{role, content}]
const labelMap = {}; // filename -> nhãn thân thiện (lấy từ sources)
let busy = false;

marked.setOptions({ breaks: true });

/* ---------- helpers ---------- */
function prettyCite(raw) {
  let s = raw.trim();
  // Nếu model lỡ ghi cả nhãn "Tư liệu N · Nguồn: X · Loại: Y" -> trích phần Nguồn/Source.
  const mSrc = s.match(/(?:Nguồn|Source)\s*:\s*([^|·\]]+)/i);
  if (mSrc) s = mSrc[1].trim();
  // Map trực tiếp từ nguồn đã truy hồi (chính xác nhất)
  const key = s.replace(/\.(md|pdf|docx?|json)$/i, "");
  if (labelMap[s]) return labelMap[s];
  if (labelMap[key]) return labelMap[key];
  // Fallback: nhận diện mã văn bản luật / dọn tên file
  let m = key.match(/^(\d+)_(\d+)_ND-CP/);
  if (m) return `Nghị định ${m[1]}/${m[2]}/NĐ-CP`;
  m = key.match(/^(\d+)_(\d+)_QH(\d+)/);
  if (m) return `Luật ${m[1]}/${m[2]}/QH${m[3]}`;
  if (/\.(md|pdf|docx?|json)$/i.test(s) || /^\d+-/.test(key)) {
    return key.replace(/^\d+-/, "").replace(/[-_]/g, " ").trim().slice(0, 70);
  }
  return s;
}

function withCitations(md) {
  // [Nguồn] -> chip đẹp, nhưng không đụng tới link [text](url)
  return md.replace(/\[([^\]\n]+)\](?!\()/g, (_, inner) =>
    `<span class="cite">${prettyCite(inner)}</span>`
  );
}
function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(withCitations(text)), { ADD_ATTR: ["class"] });
}
function scrollDown() {
  chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
}
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

/* ---------- message blocks ---------- */
function addUser(text) {
  const node = el(`<div class="msg user"><div class="role">Bạn</div><div class="bubble"></div></div>`);
  node.querySelector(".bubble").textContent = text;
  chat.appendChild(node);
  scrollDown();
}

function addAssistant() {
  const node = el(`
    <div class="msg assistant">
      <div class="role">Trợ lý</div>
      <div class="answer"><div class="typing"><span></span><span></span><span></span></div></div>
    </div>`);
  chat.appendChild(node);
  scrollDown();
  return node;
}

function tagClass(t) { return t === "legal" ? "legal" : "news"; }

function renderSources(node, via, sources) {
  if (!sources || !sources.length) return;
  const wrap = el(`
    <details class="sources">
      <summary>📚 ${sources.length} nguồn đã dùng <span class="badge">${via}</span></summary>
      <div class="source-grid"></div>
    </details>`);
  const grid = wrap.querySelector(".source-grid");
  sources.forEach((s) => {
    const pct = Math.max(4, Math.min(100, Math.round((s.score <= 1 ? s.score : 1) * 100)));
    const card = el(`
      <div class="source-card">
        <div class="source-head">
          <span class="source-name"></span>
          <span class="tag ${tagClass(s.type)}">${s.type}</span>
        </div>
        <div class="source-file"></div>
        <div class="source-snippet"></div>
        <div class="score-bar"><i style="width:${pct}%"></i></div>
      </div>`);
    card.querySelector(".source-name").textContent = s.label || s.source;
    card.querySelector(".source-file").textContent = s.source;
    card.querySelector(".source-snippet").textContent = s.snippet + "…";
    grid.appendChild(card);
  });
  node.appendChild(wrap);
}

/* ---------- streaming request ---------- */
async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true; sendBtn.disabled = true;
  if (hero) hero.remove();

  addUser(text);
  const historyToSend = messages.slice();
  messages.push({ role: "user", content: text });

  const node = addAssistant();
  const answerEl = node.querySelector(".answer");

  let acc = "";
  let firstToken = true;
  let sources = [], via = "hybrid";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: historyToSend,
        use_reranking: rerank.checked,
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "sources") {
          sources = ev.sources; via = ev.via;
          sources.forEach((s) => {
            if (s.label) {
              labelMap[s.source] = s.label;
              labelMap[s.source.replace(/\.(md|pdf|docx?|json)$/i, "")] = s.label;
            }
          });
        }
        else if (ev.type === "token") {
          if (firstToken) { answerEl.innerHTML = ""; firstToken = false; }
          acc += ev.text;
          answerEl.innerHTML = renderMarkdown(acc);
          scrollDown();
        }
      }
    }
  } catch (e) {
    answerEl.innerHTML = `<p style="color:var(--seal)">Lỗi kết nối: ${e.message}</p>`;
  }

  if (firstToken) answerEl.innerHTML = renderMarkdown(acc || "Không có phản hồi.");
  renderSources(node, via, sources);
  messages.push({ role: "assistant", content: acc });
  scrollDown();

  busy = false; sendBtn.disabled = false;
  input.focus();
}

/* ---------- events ---------- */
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = ""; autosize();
  ask(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize);

document.getElementById("chips").addEventListener("click", (e) => {
  const btn = e.target.closest(".chip");
  if (btn) ask(btn.dataset.q);
});

/* ---------- drawer: kho tài liệu ---------- */
const drawer = document.getElementById("drawer");
const libBtn = document.getElementById("libBtn");
let corpusLoaded = false;

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function domainOf(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; }
}

async function loadCorpus() {
  const body = document.getElementById("corpusBody");
  try {
    const data = await (await fetch("/api/corpus")).json();
    document.getElementById("corpusCount").textContent =
      `${data.legal.length} văn bản pháp luật · ${data.news.length} bài báo`;

    const legal = data.legal.map((d) => `
      <div class="corpus-item">
        <div class="ci-top"><span class="ci-mark">§</span><span class="ci-name">${esc(d.label)}</span></div>
        <div class="ci-file">${esc(d.file)}</div>
      </div>`).join("");

    const news = data.news.map((d) => `
      <a class="corpus-item" href="${esc(d.url)}" target="_blank" rel="noopener">
        <div class="ci-top"><span class="ci-name">${esc(d.title)}</span></div>
        <div class="ci-meta">
          <span>${esc(domainOf(d.url))}</span>
          ${d.date ? `<span>·</span><span>${esc(d.date)}</span>` : ""}
          ${d.author ? `<span>·</span><span>${esc(d.author)}</span>` : ""}
        </div>
      </a>`).join("");

    body.innerHTML = `
      <section class="corpus-group">
        <h3>Văn bản pháp luật <span class="count">${data.legal.length}</span></h3>
        ${legal}
      </section>
      <section class="corpus-group">
        <h3>Bài báo <span class="count">${data.news.length}</span></h3>
        ${news}
      </section>`;
    corpusLoaded = true;
  } catch (e) {
    body.innerHTML = `<div class="corpus-loading">Không tải được danh sách: ${esc(e.message)}</div>`;
  }
}

function openDrawer() { drawer.hidden = false; if (!corpusLoaded) loadCorpus(); }
function closeDrawer() { drawer.hidden = true; }

libBtn.addEventListener("click", openDrawer);
drawer.addEventListener("click", (e) => { if (e.target.dataset.close !== undefined) closeDrawer(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) closeDrawer(); });

input.focus();
