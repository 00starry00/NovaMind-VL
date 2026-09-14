/* ============================================================
   星眸 XingMou · 前端逻辑
   ============================================================ */
"use strict";

const $ = (id) => document.getElementById(id);
const chatEl = $("chat"), inputEl = $("input"), sessionListEl = $("sessionList");

const state = {
  sessionId: "",            // 当前会话 id（""=新会话）
  streaming: false,
  abort: null,              // AbortController
  imageMode: false,         // 👁 强制看图开关：ON 时有图会话所有问题都看图；OFF 时自动判断
  sessionHasImage: false,   // 当前会话是否已看过图
  settings: {
    temperature: 0.7, top_p: 0.9, top_k: 50, max_new_tokens: 256,
  },
  pendingImage: null,       // { b64, previewUrl }
};

/* ---------------- 星空背景（星斗 + 偶发流星）---------------- */
(function stars() {
  const cv = $("stars"), ctx = cv.getContext("2d");
  let stars = [], meteors = [];
  function resize() {
    cv.width = innerWidth; cv.height = innerHeight;
    stars = Array.from({ length: 190 }, () => ({
      x: Math.random() * cv.width, y: Math.random() * cv.height,
      r: Math.random() * 1.4 + 0.3,
      a: Math.random() * Math.PI * 2, s: Math.random() * 0.02 + 0.004,
      big: Math.random() < 0.06,
    }));
  }
  resize(); addEventListener("resize", resize);
  // 偶发流星
  setInterval(() => {
    if (meteors.length < 2) {
      meteors.push({
        x: Math.random() * cv.width * 0.8 + cv.width * 0.2,
        y: Math.random() * cv.height * 0.3,
        vx: -(Math.random() * 5 + 4), vy: Math.random() * 2.5 + 1.5,
        life: 1,
      });
    }
  }, 6000);
  (function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (const st of stars) {
      st.a += st.s;
      const o = 0.35 + 0.65 * Math.abs(Math.sin(st.a));
      ctx.beginPath();
      ctx.arc(st.x, st.y, st.big ? st.r * 2.2 : st.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${st.big ? "245,198,107" : "200,215,255"},${o})`;
      ctx.fill();
      if (st.big) {
        ctx.beginPath();
        ctx.arc(st.x, st.y, st.r * 5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(245,198,107,${o * 0.12})`;
        ctx.fill();
      }
    }
    meteors = meteors.filter((m) => m.life > 0);
    for (const m of meteors) {
      m.x += m.vx; m.y += m.vy; m.life -= 0.012;
      const grad = ctx.createLinearGradient(m.x, m.y, m.x - m.vx * 14, m.y - m.vy * 14);
      grad.addColorStop(0, `rgba(255,255,255,${m.life})`);
      grad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(m.x, m.y);
      ctx.lineTo(m.x - m.vx * 14, m.y - m.vy * 14);
      ctx.stroke();
    }
    requestAnimationFrame(draw);
  })();
})();

/* ---------------- 工具 ---------------- */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function mdLite(text) {
  let t = esc(text || "");
  const blocks = [];
  t = t.replace(/```([\s\S]*?)```/g, (m, code) => {
    blocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `\u0000B${blocks.length - 1}\u0000`;
  });
  t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  // 工具调用标记 → 金色小芯片（纯 CSS 图标，不用 emoji 避免方块）
  t = t.replace(/🔧([\w_]+)/g, '<span class="tool-chip">$1</span>');
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  t = t.replace(/(^|\n)(#{1,3})\s+(.+)/g, (m, nl, h, txt) => `${nl}<h${h.length}>${txt}</h${h.length}>`);
  t = t.replace(/(^|\n)&gt;\s?(.+)/g, '$1<blockquote>$2</blockquote>');
  t = t.replace(/(^|\n)[*-]\s+(.+)/g, "$1<ul><li>$2</li></ul>");
  t = t.replace(/(^|\n)\d+\.\s+(.+)/g, "$1<ol><li>$2</li></ol>");
  t = t.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank">$1</a>');
  t = t.split("\n").join("<br>");
  t = t.replace(/\u0000B(\d+)\u0000/g, (m, i) => blocks[+i]);
  return t;
}

/* ---------------- 消息渲染 ---------------- */
function addMsg(role, text, opts = {}) {
  $("welcome")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const isUser = role === "user";
  const starSvg = '<svg viewBox="0 0 24 24" width="15" height="15"><path d="M12 1.5 L14.8 9.2 L22.5 12 L14.8 14.8 L12 22.5 L9.2 14.8 L1.5 12 L9.2 9.2 Z" fill="currentColor"/></svg>';
  div.innerHTML = `
    <div class="avatar">${isUser ? "我" : starSvg}</div>
    <div class="body">
      ${opts.image ? `<img class="msg-img" src="${opts.image}" alt="图片">` : ""}
      <div class="bubble">${isUser ? esc(text) : ""}</div>
      ${opts.meta ? `<div class="meta">${opts.meta}</div>` : ""}
    </div>`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addUser(text, imageUrl) {
  addMsg("user", text.replace("<|image|>", ""), { image: imageUrl });
}

function addAssistant(opts = {}) {
  const div = addMsg("assistant", "");
  const bubble = div.querySelector(".bubble");
  return {
    div, bubble,
    setText: (t) => { bubble.innerHTML = mdLite(t); },
    setMeta: (m) => {
      let metaEl = div.querySelector(".meta");
      if (!metaEl) { metaEl = document.createElement("div"); metaEl.className = "meta"; div.querySelector(".body").appendChild(metaEl); }
      metaEl.textContent = m;
    },
  };
}

/* ---------------- SSE 请求 ---------------- */
async function streamRequest(path, body, onDelta, onDone, onError) {
  state.abort = new AbortController();
  const t0 = performance.now();
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: state.abort.signal,
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "delta") onDelta(ev.text);
          else if (ev.type === "done") onDone(ev);
          else if (ev.type === "error") onError(new Error(ev.message));
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") onError(e);
  } finally {
    state.streaming = false;
    state.abort = null;
    $("btnSend").disabled = false;
    $("btnSend").textContent = "发送";
    $("topStatus")?.classList.add("hidden");   // 思考中指示归位
  }
}

/* ---------------- 发送 ---------------- */
async function send() {
  const text = inputEl.value.trim();
  if (!text || state.streaming) return;
  sendMessage(text);
}

async function sendMessage(text) {
  const img = state.pendingImage;
  state.pendingImage = null;
  $("imagePreview").classList.add("hidden");
  fileInput.value = "";
  state.streaming = true;
  $("btnSend").disabled = true; $("btnSend").textContent = "停止";
  inputEl.value = ""; inputEl.style.height = "auto";
  addUser(text || "看看这张图", img?.previewUrl);
  const a = addAssistant();
  // 智能体思考状态：头像呼吸 + 三点跳动，首个字到达后切换成打字光标
  a.div.classList.add("thinking");
  a.bubble.innerHTML = '<span class="think-dots"><i></i><i></i><i></i></span>';
  $("topStatus")?.classList.remove("hidden");
  let started = false;
  const acc = [];
  const body = {
    session_id: state.sessionId,
    message: text || "描述这张图片",
    force_vision: state.imageMode,
    ...state.settings,
  };
  if (img) body.image_b64 = img.b64;
  streamRequest("/api/chat/stream", body,
    (d) => {
      if (!started) {
        started = true;
        a.div.classList.remove("thinking");
        a.bubble.innerHTML = "";
      }
      acc.push(d); a.setText(acc.join("")); a.bubble.innerHTML += '<span class="caret"></span>'; chatEl.scrollTop = chatEl.scrollHeight;
    },
    (ev) => {
      a.bubble.querySelector(".caret")?.remove();
      a.setText(ev.reply);
      a.setMeta(`${ev.mode === "vision" ? "识图模式" : "文字模式"} · ${ev.speed} tok/s`);
      adoptSession(ev.session_id);
      if (ev.has_image) state.sessionHasImage = true;
      syncModeChip();
      refreshSessions();
    },
    (e) => { a.setMeta(`出错: ${e.message}`); });
}

$("btnSend").addEventListener("click", () => {
  if (state.streaming) { state.abort?.abort(); return; }
  send();
});

/* ---------------- 看图开关（强制看图） ---------------- */
function syncModeChip() {
  $("btnVision").classList.toggle("on", state.imageMode);
  $("modeChip").classList.toggle("hidden", !state.imageMode);
}
$("btnVision").addEventListener("click", () => {
  if (!state.imageMode && !state.sessionHasImage && !state.pendingImage) {
    $("fileInput").click();   // 没图先选图
    return;
  }
  state.imageMode = !state.imageMode;   // 开启后强制带图回答；关闭后自动判断
  syncModeChip();
  inputEl.focus();
});

/* 能力卡片：点击聚焦输入框 / 唤起选图（智能体技能入口） */
function bindCaps() {
  $("capChat")?.addEventListener("click", () => inputEl.focus());
  $("capVision")?.addEventListener("click", () => $("fileInput").click());
}

/* ---------------- 会话管理 ---------------- */
function adoptSession(sid) {
  if (sid && sid !== state.sessionId) state.sessionId = sid;
}

async function refreshSessions() {
  try {
    const r = await fetch("/api/sessions");
    const { sessions } = await r.json();
    sessionListEl.innerHTML = "";
    for (const s of sessions) {
      const item = document.createElement("div");
      item.className = "session-item" + (s.id === state.sessionId ? " active" : "");
      item.innerHTML = `
        <span class="s-icon ${s.has_image ? "vision" : ""}">${s.has_image ? "▧" : "◎"}</span>
        <span class="s-title">${esc(s.title)}</span>
        <button class="s-del" title="删除">✕</button>`;
      item.addEventListener("click", (e) => {
        if (e.target.classList.contains("s-del")) return;
        openSession(s.id);
      });
      item.querySelector(".s-del").addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
        if (s.id === state.sessionId) newSession();
        refreshSessions();
      });
      sessionListEl.appendChild(item);
    }
  } catch {}
}

async function openSession(sid) {
  state.sessionId = sid;
  state.streaming && state.abort?.abort();
  chatEl.innerHTML = "";
  $("welcome")?.remove();
  try {
    const r = await fetch(`/api/sessions/${sid}/messages`);
    const data = await r.json();
    if (data.error) return;
    $("modeChip").classList.toggle("hidden", !data.has_image);
    state.sessionHasImage = !!data.has_image;
    state.imageMode = false;   // 恢复会话不自动进入强制看图，服务端自动路由
    syncModeChip();
    $("topTitle").textContent = data.title || "星眸";
    for (const m of data.messages) {
      if (m.role === "user") {
        const isImg = m.content.includes("<|image|>");
        const text = m.content.replace("<|image|>\n", "");
        const div = document.createElement("div");
        div.className = "msg user";
        div.innerHTML = `
          <div class="avatar">我</div>
          <div class="body">
            ${isImg ? '<div class="img-tag">▧ 图片</div>' : ""}
            <div class="bubble">${esc(text)}</div>
          </div>`;
        chatEl.appendChild(div);
      } else {
        const a = addAssistant();
        a.setText(m.content);
        a.setMeta(data.has_image ? "识图模式" : "文字模式");
      }
    }
  } catch {}
  refreshSessions();
}

function newSession() {
  state.streaming && state.abort?.abort();
  state.sessionId = "";
  state.pendingImage = null;
  state.imageMode = false;
  state.sessionHasImage = false;
  $("imagePreview").classList.add("hidden");
  syncModeChip();
  $("topTitle").textContent = "星眸";
  chatEl.innerHTML = "";
  chatEl.innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome-star">✦</div>
      <h1>星眸 <span>XingMou</span></h1>
      <p class="welcome-sub">由日落星辉开发 · 0.3B 多模态小模型</p>
      <div class="feature-pills">
        <button class="cap-card" id="capChat">
          <span class="cap-icon">
            <svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 4h16v12H8l-4 4z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><circle cx="9" cy="10" r="0.9" fill="currentColor"/><circle cx="12.6" cy="10" r="0.9" fill="currentColor"/><circle cx="16.2" cy="10" r="0.9" fill="currentColor"/></svg>
          </span>
          <span class="cap-name">聊天</span>
          <span class="cap-desc">问答 · 常识 · 闲聊</span>
        </button>
        <button class="cap-card" id="capVision">
          <span class="cap-icon">
            <svg viewBox="0 0 24 24" width="22" height="22"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.7"/><circle cx="12" cy="12" r="2.6" fill="currentColor"/></svg>
          </span>
          <span class="cap-name">识图</span>
          <span class="cap-desc">看图 · 描述 · 追问</span>
        </button>
      </div>
      <div class="examples" id="examples"></div>
      <p class="welcome-note">能力：聊天 + 识图 · 0.3B 小模型，回答仅供参考</p>
    </div>`;
  bindExamples();
  bindCaps();
  refreshSessions();
  inputEl.focus();
}

function bindExamples() {
  const ex = ["你是谁？", "1+1 等于几？", "写一首关于星星的诗",
              "给我讲个小故事", "什么是机器学习？"];
  const wrap = $("examples");
  if (!wrap) return;
  wrap.innerHTML = "";
  for (const t of ex) {
    const b = document.createElement("button");
    b.className = "ex-chip"; b.textContent = t;
    b.addEventListener("click", () => { inputEl.value = t; inputEl.focus(); });
    wrap.appendChild(b);
  }
  const imgBtn = document.createElement("button");
  imgBtn.className = "ex-chip"; imgBtn.textContent = "▧ 上传一张图片让我看看";
  imgBtn.addEventListener("click", () => $("fileInput").click());
  wrap.appendChild(imgBtn);
}

$("btnNew").addEventListener("click", newSession);

/* ---------------- 图片输入 ---------------- */
const fileInput = $("fileInput");
function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    state.pendingImage = { b64: dataUrl.split(",")[1], previewUrl: dataUrl };
    // 不再自动开启强制看图：发送带图消息自动识图，后续问题由服务端自动路由
    syncModeChip();
    $("imageThumb").src = dataUrl;
    $("imagePreview").classList.remove("hidden");
    inputEl.focus();
  };
  reader.readAsDataURL(file);
}
$("btnAttach").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
$("imgRemove").addEventListener("click", () => {
  state.pendingImage = null;
  $("imagePreview").classList.add("hidden");
  fileInput.value = "";
});

// 拖拽
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
// 粘贴
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) handleFile(item.getAsFile());
});

/* ---------------- 输入框 ---------------- */
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

/* ---------------- 设置 ---------------- */
function bindSettings() {
  const map = [
    ["setTemp", "temperature"], ["setTopP", "top_p"], ["setTopK", "top_k"], ["setMaxLen", "max_new_tokens"],
  ];
  for (const [elId, key] of map) {
    const el = $(elId);
    el.value = state.settings[key];
    const sync = () => {
      state.settings[key] = parseFloat(el.value);
      $(elId + "Val").textContent = el.value;
    };
    el.addEventListener("input", sync); sync();
  }
}
$("btnSettings").addEventListener("click", () => $("settingsPop").classList.toggle("hidden"));
$("btnCloseSettings").addEventListener("click", () => $("settingsPop").classList.add("hidden"));

/* ---------------- 启动 ---------------- */
(async function init() {
  bindSettings();
  bindExamples();
  bindCaps();
  try {
    const h = await fetch("/health").then((r) => r.json());
    const chip = $("statusChip");
    if (h.ok) {
      chip.classList.add("ok");
      chip.innerHTML =
        `<span class="chip-avatar">✦</span>` +
        `<span class="chip-text"><b>星眸 · 在线</b>` +
        `<i>${h.device === "cuda" ? "GPU 推理" : "CPU 运行"} · 显存 ${h.vram_gb ?? "?"}G · ${h.models.name}</i></span>`;
    } else {
      chip.innerHTML = `<span class="dot"></span>服务异常`;
    }
  } catch {
    $("statusChip").innerHTML = `<span class="dot"></span>服务未启动`;
  }
  refreshSessions();
})();
