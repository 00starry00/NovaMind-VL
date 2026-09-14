"""
================================================================================
agent.py —— NovaMind-VL 本地推理核心（打包后的"智能体大脑"）
================================================================================

职责：
  1. 加载两套权重：
       - 文本模型  sft_clean_0.3b.pth   （纯文字聊天，SFT 版，无 DPO）
       - VLM 模型  vlm_s2v2_0.3b.pth    （词表 32001，看图对话，SigLIP2）
       - SigLIP2 视觉塔 + 投影层 vlm_s2v2_0.3b_projector.pt

【2026-09-13 回退决策】
  用户实测 DPO + 工具循环决策反而让聊天体验退化（过度触发工具、JSON 污染
  回复、罐头残留），决定回退：
       - 文字模型：sft_clean_0.3b（DPO 之前的版本，身份完好）
       - 工具循环：停用（chat_stream_tools / execute_tool 代码保留但不再
         加载 tool3 模型、不再进入决策循环，省 ~1.3G 显存）
       - 识图：vlm_s2v2_0.3b + SigLIP2（不变）
  若日后要恢复工具模式：重新加载 tool_model + enable_tools=True，
  server.py 的 /api/chat/stream 改回条件路由即可。
  2. 会话管理：多轮历史、文字/看图路由（带图会话自动切 VLM）
  3. 流式生成：temperature/top-k/top-p + 重复惩罚 + KV cache

路由规则（2026-09-13 自动路由版）：
  - 带新图的消息        → 强制看图（VLM + SigLIP2，<|image|> 占位在视觉上下文开头）
  - 会话里有图、无新图  → 自动判断：问题含图相关词/代词（她他它）、或超短追问 → 看图；
                         否则 → 纯文字（sft_clean + 完整会话记忆）
  - 会话里没有图        → 纯文字
  - 👁 开关开启（force_vision）→ 有图会话所有问题都看图
  - 发新图不清空会话记忆：文字历史全保留；旧图的问答不进新图的 VLM 上下文（防串图）

被 serve/server.py 调用；也可以直接 python agent.py 自测。
================================================================================
"""

import base64
import io
import json
import os
import re
import threading
import time
from collections import OrderedDict

import torch
from PIL import Image
from transformers import AutoTokenizer

# serve/ 的上一级是项目根，novamind 包在根下
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, _ROOT)

from novamind import ModelConfig, PRESETS, NovaMindForCausalLM
from novamind.vision_model import NovaMindVision
from novamind.vlm_utils import resize_llm_vocab, splice_image_features

MODEL_STORE = os.environ.get(
    "NOVAMIND_MODEL_STORE",
    os.path.join(_ROOT, "model_store"),
)
SESSIONS_DIR = os.environ.get(
    "NOVAMIND_SESSIONS_DIR",
    os.path.join(MODEL_STORE, "sessions"),
)
WORKSPACE = os.environ.get(
    "NOVAMIND_WORKSPACE",
    os.path.join(_ROOT, "serve", "workspace"),
)

# 工具清单（拼进文字会话的 system 提示，模型按清单调用）
TOOL_SYSTEM = (
    "（可用工具：\n"
    "- get_time: 查询当前日期和时间\n"
    "- create_file: 新建文件，参数为文件名\n"
    "- create_folder: 新建文件夹，参数为文件夹名\n"
    "- read_file: 读取文件内容，参数为文件名\n"
    "- list_dir: 列出目录内容，参数为目录名，空字符串为根目录\n"
    "需要执行操作时，只输出一行 JSON，例如 {\"tool\": \"create_folder\", \"args\": \"rui\"}。\n"
    "重要：没有合适的工具（比如天气、搜索、上网）就直接用中文回答，"
    "绝对不要编造工具调用、编造工具结果或伪造数据。\n）"
)
MAX_TOOL_ROUNDS = 3

# 图相关词/代词（自动路由用：命中 → 看图；不命中且问题较长 → 纯文字）
_IMG_KW = re.compile(
    r"图片|照片|图中|图里|图像|截图|画面|看图|识图|这张|那张|上图|下图|"
    r"背景|场景|主体|颜色|什么色|衣服|裙子|头发|眼睛|表情|牌子|标志|"
    r"写的字|上面写|好看|漂亮|穿的|戴的|戴着|他|她|它"
)


def execute_tool(tool, args):
    """真实执行工具（沙盒：WORKSPACE 目录内），返回结果字符串。"""
    if tool == "get_time":
        if args and str(args).strip():
            # 模型把不该查时间的问题误判成 get_time（如问天气）→ 明确纠正
            return (f"get_time 不接受参数。用户问的是「{str(args).strip()}」，"
                    f"你没有天气/搜索/联网工具，请如实回答做不到，不要编造结果。")
        return time.strftime("%Y年%m月%d日 %H:%M:%S（%A）")
    if tool == "create_file":
        name = str(args or "").strip() or "未命名.txt"
        os.makedirs(WORKSPACE, exist_ok=True)
        p = os.path.join(WORKSPACE, os.path.basename(name))
        with open(p, "w", encoding="utf-8") as f:
            f.write("")
        return f"已创建文件 {os.path.basename(name)}（位于星眸工作区）"
    if tool == "create_folder":
        name = str(args or "").strip() or "新文件夹"
        os.makedirs(os.path.join(WORKSPACE, os.path.basename(name)), exist_ok=True)
        return f"已创建文件夹 {os.path.basename(name)}（位于星眸工作区）"
    if tool == "read_file":
        name = str(args or "").strip()
        p = os.path.join(WORKSPACE, os.path.basename(name))
        if not os.path.exists(p):
            return f"文件 {name} 不存在"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return ("（文件内容）" + content) if content.strip() else "（文件是空的）"
    if tool == "list_dir":
        sub = str(args or "").strip()
        if sub in ("", ".", "/", "工作区", "根目录", "工作目录", "workspace"):
            d = WORKSPACE
        else:
            d = os.path.join(WORKSPACE, sub)
        if not os.path.isdir(d):
            return f"目录 {sub or '根目录'} 不存在"
        entries = sorted(os.listdir(d))
        if not entries:
            return f"目录 {sub or '根目录'} 是空的"
        return f"目录 {sub or '根目录'} 下有：" + "、".join(entries)
    return (f"工具 {tool} 不存在。请如实告诉用户：你只有查时间、文件读写、"
            f"目录列表这几个能力，没有这个工具，不要编造结果。")


# --------------------------------------------------------------------------- #
#  会话
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self):
        self.messages = []        # [{role, content}, ...]
        self.has_image = False
        self.image = None         # PIL 图（看图的会话里保留）
        self.vision_start = 0     # 当前图视觉上下文在 messages 里的起点（新图时重置）
        self.title = "新会话"
        self.created = time.time()
        self.updated = time.time()


# --------------------------------------------------------------------------- #
#  加载
# --------------------------------------------------------------------------- #
def _load_llm(weight_path, tokenizer_dir, device, dtype):
    """加载一个 LLM 权重 + 分词器，自动按权重词表建模型。"""
    state = torch.load(weight_path, map_location="cpu")
    vocab = state["model.embed_tokens.weight"].shape[0]
    cfg = ModelConfig(**PRESETS["novamind-0.3b"])
    cfg.vocab_size = vocab
    model = NovaMindForCausalLM(cfg)
    model.load_state_dict(state)
    model.to(device).to(dtype).eval()
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tok


class NovaMindAgent:
    """打包后的 NovaMind-VL 推理核心：文字 + 识图，多会话。"""

    def __init__(self, model_store=MODEL_STORE, device=None, dtype=torch.float16,
                 max_sessions=50, max_history=20, max_new_tokens=256,
                 temperature=0.7, top_k=50, top_p=0.9,
                 persona=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_sessions = max_sessions
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # 新会话第一句话前拼的人设（环境变量 NOVAMIND_PERSONA 可覆盖，置空关闭）
        # ⚠️ 人设里不要带身份词（星眸/NovaMind/日落星辉），会触发小模型的自我介绍罐头（踩坑 #12）
        self.persona = (persona if persona is not None
                        else os.environ.get(
                            "NOVAMIND_PERSONA",
                            "回答要简洁友好，用中文，直接回答用户的问题。"))
        self._persona_applied = set()

        out_dir = os.path.join(model_store, "out")
        text_weight = os.environ.get("NOVAMIND_TEXT_WEIGHT", "sft_clean_0.3b")
        # 工具决策循环：已回退停用（2026-09-13），不再加载工具模型
        self.enable_tools = False
        vlm_weight = os.environ.get("NOVAMIND_VLM_WEIGHT", "vlm_s2v2_0.3b")
        vlm_proj = os.environ.get("NOVAMIND_VLM_PROJECTOR", "vlm_s2v2_0.3b_projector")
        vision_dir = os.environ.get("NOVAMIND_VISION", "siglip2-256")
        t0 = time.time()
        print(f"[agent] 加载文本模型 ({text_weight}) ...")
        self.text_model, self.text_tok = _load_llm(
            os.path.join(out_dir, f"{text_weight}.pth"),
            os.path.join(model_store, "tokenizer"),
            self.device, self.dtype)
        # 工具模型（tool3）：已回退停用（2026-09-13），不加载，省显存。
        # 恢复方法见文件顶部【2026-09-13 回退决策】。
        self.tool_model = None
        print(f"[agent] 加载 VLM 模型 ({vlm_weight}) ...")
        self.vlm_model, self.vlm_tok = _load_llm(
            os.path.join(out_dir, f"{vlm_weight}.pth"),
            os.path.join(model_store, "vlm_tokenizer"),
            self.device, self.dtype)
        if self.vlm_model.config.vocab_size < 32001:
            resize_llm_vocab(self.vlm_model, self.vlm_model.config.vocab_size, 32001)

        print(f"[agent] 加载视觉塔 ({vision_dir}) ...")
        self.vision = NovaMindVision(
            os.path.join(model_store, vision_dir),
            llm_hidden=self.vlm_model.config.hidden_size)
        proj = torch.load(os.path.join(out_dir, f"{vlm_proj}.pt"), map_location="cpu")
        self.vision.projector.load_state_dict(proj)
        self.vision.to(self.device).to(self.dtype).eval()
        self.vlm_tag = f"{vlm_weight} ({vision_dir.replace('-vision-only', '')})"

        self.sessions = OrderedDict()
        self.lock = threading.Lock()          # GPU 生成串行化
        self.loaded = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1e9) if self.device == "cuda" else 0
        self._restore_sessions()              # 从磁盘恢复会话（重启不丢记忆）
        print(f"[agent] 就绪 device={self.device} 加载耗时 {self.loaded:.1f}s "
              f"显存 {vram:.2f}G 会话 {len(self.sessions)}")

    # ------------------------------------------------------------------ #
    #  会话持久化（存磁盘，服务重启不丢）
    # ------------------------------------------------------------------ #
    def _persist(self):
        try:
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            for sid, s in self.sessions.items():
                data = {"messages": s.messages, "has_image": s.has_image,
                        "vision_start": s.vision_start,
                        "title": s.title, "created": s.created,
                        "updated": s.updated,
                        "persona_applied": sid in self._persona_applied}
                with open(os.path.join(SESSIONS_DIR, f"{sid}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                if s.image is not None:
                    try:
                        s.image.save(os.path.join(SESSIONS_DIR, f"{sid}.jpg"))
                    except Exception:
                        pass
        except Exception as e:
            print(f"[agent] 会话保存失败: {e}")

    def _restore_sessions(self):
        import glob as _glob
        try:
            if not os.path.isdir(SESSIONS_DIR):
                return
            # 启动时自动备份会话目录（防误删事故，保留最近 5 份）
            try:
                backup_dir = os.path.join(SESSIONS_DIR, "..", "sessions_backup")
                os.makedirs(backup_dir, exist_ok=True)
                name = f"sessions_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
                import tarfile as _tarfile
                with _tarfile.open(os.path.join(backup_dir, name), "w:gz") as tf:
                    tf.add(SESSIONS_DIR, arcname="sessions")
                olds = sorted(os.listdir(backup_dir))
                for old in olds[:-5]:
                    try:
                        os.remove(os.path.join(backup_dir, old))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[agent] 会话备份失败: {e}")
            for f in _glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
                try:
                    sid = os.path.splitext(os.path.basename(f))[0]
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    s = Session()
                    s.messages = data.get("messages", [])
                    s.has_image = data.get("has_image", False)
                    s.vision_start = data.get("vision_start", 0)
                    s.title = data.get("title", "新会话")
                    s.created = data.get("created", time.time())
                    s.updated = data.get("updated", s.created)
                    if data.get("persona_applied"):
                        self._persona_applied.add(sid)
                    img_path = os.path.join(SESSIONS_DIR, f"{sid}.jpg")
                    if s.has_image and os.path.exists(img_path):
                        try:
                            s.image = Image.open(img_path).convert("RGB")
                        except Exception:
                            pass
                    self.sessions[sid] = s
                except Exception:
                    continue
            print(f"[agent] 已恢复 {len(self.sessions)} 个会话")
        except Exception as e:
            print(f"[agent] 会话恢复失败: {e}")

    # ------------------------------------------------------------------ #
    #  会话
    # ------------------------------------------------------------------ #
    def _apply_persona(self, session_id, s, message):
        """每个会话只拼一次人设（首条用户消息前）。"""
        if self.persona and session_id not in self._persona_applied:
            self._persona_applied.add(session_id)
            return f"{self.persona}\n\n{message}"
        return message

    def get_session(self, session_id):
        if not session_id:
            session_id = "s_%d" % int(time.time() * 1000)
        s = self.sessions.get(session_id)
        if s is None:
            if len(self.sessions) >= self.max_sessions:
                self.sessions.popitem(last=False)   # 淘汰最旧
            s = Session()
            self.sessions[session_id] = s
        s.updated = time.time()
        return session_id, s

    def _set_title(self, s, raw_message):
        """首条消息自动起标题（去掉 <|image|>，截 18 字）。"""
        if s.title != "新会话":
            return
        t = raw_message.replace("<|image|>", "").strip()
        if self.persona and t.startswith(self.persona):
            t = t[len(self.persona):].strip()
        s.title = (t[:18] + "…") if len(t) > 18 else (t or "新会话")

    def list_sessions(self):
        out = []
        for sid, s in self.sessions.items():
            out.append({
                "id": sid, "title": s.title, "has_image": s.has_image,
                "n_messages": len(s.messages),
                "updated": s.updated, "created": s.created,
            })
        return sorted(out, key=lambda x: x["updated"], reverse=True)

    def get_messages(self, session_id):
        s = self.sessions.get(session_id)
        if s is None:
            return None
        return {"messages": list(s.messages), "has_image": s.has_image,
                "title": s.title}

    def clear_session(self, session_id):
        self.sessions.pop(session_id, None)
        # 防护（2026-09-13）：删除改为移入 trash 目录，不直接删文件，
        # 防止测试/误操作导致会话记录永久丢失（曾发生批量误删事故）
        try:
            trash = os.path.join(SESSIONS_DIR, "trash")
            os.makedirs(trash, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for ext in (".json", ".jpg"):
                p = os.path.join(SESSIONS_DIR, f"{session_id}{ext}")
                if os.path.exists(p):
                    os.replace(p, os.path.join(trash, f"{stamp}_{session_id}{ext}"))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  图片
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_image(image_path=None, image_b64=None):
        """从路径或 base64 读 PIL 图（都失败返回 None）。"""
        if image_path:
            try:
                return Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"图片读取失败: {e}")
        if image_b64:
            try:
                raw = base64.b64decode(image_b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise ValueError(f"base64 图片解码失败: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  生成（流式）
    # ------------------------------------------------------------------ #
    def _encode_image(self, image):
        pixel = self.vision.image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel = pixel.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            return self.vision(pixel)          # [1, N, 1024]

    @torch.no_grad()
    def _generate(self, model, tok, ids, img_feats=None, img_pos=-1,
                  max_new_tokens=None, temperature=None, top_k=None, top_p=None):
        """自回归生成，yield 文本增量。ids 是完整 prompt。采样参数可临时覆盖。"""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p

        input_ids = torch.tensor([ids], device=self.device)
        labels = torch.full_like(input_ids, -100)
        attn = torch.ones_like(input_ids)

        if img_feats is not None and img_pos >= 0:
            pos_t = torch.tensor([img_pos], device=self.device)
            embeds, _, attn = splice_image_features(
                model.model.embed_tokens, input_ids, labels, attn, img_feats, pos_t)
            out = model(inputs_embeds=embeds, attention_mask=attn,
                        use_cache=True, logits_to_keep=1)
        else:
            out = model(input_ids=input_ids, attention_mask=attn,
                        use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        generated = []
        shown = ""
        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-6)
            if generated:  # 重复惩罚
                for t in set(generated):
                    logits[0, t] = (logits[0, t] / 1.15 if logits[0, t] > 0
                                    else logits[0, t] * 1.15)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float("inf")
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                logits[0, sorted_i[0][remove[0]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()
            if nxt == tok.eos_token_id:
                break
            generated.append(nxt)

            full = tok.decode(generated)
            # 增量输出；未完整的多字节字符会解码成 \ufffd，先剥掉（下一增量会补齐），
            # 否则前端流式渲染会出现乱码方块
            delta = full[len(shown):].replace("\ufffd", "")
            if delta:
                yield delta
                shown += delta

            next_ids = torch.tensor([[nxt]], device=self.device)
            out = model(input_ids=next_ids, past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #
    def chat(self, session_id, message):
        """纯文字多轮对话，返回 (完整回复, session_id)。"""
        reply, sid = "", session_id
        for text, sid2, done in self.chat_stream(session_id, message):
            if done:
                reply, sid = text, sid2
        return reply, sid

    def _parse_tool(self, text):
        """尝试从回复里解析工具调用 JSON，返回 (tool, args) 或 None。"""
        t = text.strip()
        if not t.startswith("{"):
            return None
        try:
            obj = json.loads(t.split("\n")[0])
            if isinstance(obj, dict) and "tool" in obj:
                return obj.get("tool"), obj.get("args", "")
        except Exception:
            pass
        return None

    def chat_stream(self, session_id, message, **kw):
        """纯文字多轮对话（流式生成器）——永远走文本模型。
        看图走 see_stream，由调用方（网页的看图开关）显式选择。
        yield (文本增量, session_id, False)；最后一条 (完整回复, session_id, True)。"""
        def gen():
            with self.lock:
                session_id2, s = self.get_session(session_id)
                self._set_title(s, message)
                msg = self._apply_persona(session_id2, s, message)
                s.messages.append({"role": "user", "content": msg})
                model, tok = self.text_model, self.text_tok
                # 历史里可能残留 <|image|> 占位（之前看过图）：文本分词器不认识它，剥掉
                hist = [{"role": m["role"],
                         "content": m["content"].replace("<|image|>\n", "")
                         .replace("<|image|>", "")}
                        for m in s.messages]
                prompt = tok.apply_chat_template(
                    hist, add_generation_prompt=True, tokenize=False)
                ids = tok.encode(prompt)
                img_feats, img_pos = None, -1

                parts = []
                for d in self._generate(model, tok, ids, img_feats, img_pos, **kw):
                    parts.append(d)
                    yield d, session_id2, False
                reply = "".join(parts)
                s.messages.append({"role": "assistant", "content": reply})
                s.messages = s.messages[-self.max_history:]
                self._persist()
                yield reply, session_id2, True
        return gen()

    def chat_stream_tools(self, session_id, message, **kw):
        """带工具调用的多轮对话（工具开关打开时走这里）。
        模型可输出 {"tool": ...} → 真实执行 → 结果回填 → 继续生成（最多 3 轮）。
        yield (文本增量, session_id, False)；最后一条 (完整回复, session_id, True)。"""
        def gen():
            with self.lock:
                session_id2, s = self.get_session(session_id)
                self._set_title(s, message)
                msg = self._apply_persona(session_id2, s, message)
                s.messages.append({"role": "user", "content": msg})
                # 工具模型常驻内存；缺失时退回纯聊天模型（此时工具大概率乱调）
                model = self.tool_model if self.tool_model is not None else self.text_model
                tok = self.text_tok

                # 历史：去掉 <|image|> 残留；工具清单拼在最后一条 user 消息前（始终可见）
                hist = [{"role": m["role"],
                         "content": m["content"].replace("<|image|>\n", "")
                         .replace("<|image|>", "")}
                        for m in s.messages]
                if hist and hist[-1]["role"] == "user":
                    hist[-1]["content"] = TOOL_SYSTEM + "\n" + hist[-1]["content"]

                parts_all = []
                final_text = ""
                for _round in range(MAX_TOOL_ROUNDS):
                    prompt = tok.apply_chat_template(
                        hist, add_generation_prompt=True, tokenize=False)
                    ids = tok.encode(prompt)
                    parts = []
                    first_round = (_round == 0)
                    for d in self._generate(model, tok, ids, None, -1, **kw):
                        parts.append(d)
                        # 第一轮先缓冲：若是工具调用，不把原始 JSON 流给用户
                        if not first_round:
                            yield d, session_id2, False
                    text = "".join(parts).strip()
                    parts_all.append(text)
                    parsed = self._parse_tool(text)
                    if parsed is None:
                        final_text = text
                        if first_round:
                            # 纯聊天：补流缓冲的内容
                            for d in parts:
                                yield d, session_id2, False
                        break
                    tool, args = parsed
                    result = execute_tool(tool, args)
                    yield f" 🔧{tool}", session_id2, False   # 干净的工具调用标记
                    hist.append({"role": "assistant", "content": text})
                    hist.append({"role": "user",
                                 "content": f"工具返回：{result}"})

                reply = final_text or (parts_all[-1] if parts_all else "")
                s.messages.append({"role": "assistant", "content": reply})
                s.messages = s.messages[-self.max_history:]
                self._persist()
                yield reply, session_id2, True
        return gen()

    def see(self, session_id, question, image=None):
        """看图问答，返回 (完整回复, session_id)。"""
        reply, sid = "", session_id
        for text, sid2, done in self.see_stream(session_id, question, image):
            if done:
                reply, sid = text, sid2
        return reply, sid

    def _looks_image_related(self, question):
        """启发式：判断问题是否和会话里的图有关。
        - 命中图相关词/代词（她他它/颜色/穿的/图中…）→ 看图
        - 超短追问（≤4 字，如“？”、“好看吗”、“然后呢”）→ 看图（追问图是常态）
        - 其余（如“什么是机器学习”、“今天星期几”）→ 纯文字
        纯规则、零延迟，不赌小模型的判断力（踩坑 #14：0.3B 无法自主判断图文相关性）。"""
        q = question.strip()
        if _IMG_KW.search(q):
            return True
        if len(q) <= 4:
            return True
        return False

    def auto_stream(self, session_id, message, image=None, force_vision=False, **kw):
        """自动路由流式对话（网页统一入口）。
        - 带新图 → 看图（图放视觉上下文开头）；发新图不清空会话记忆
        - 有图无新图 → _looks_image_related 自动判断看图还是纯文字
        - force_vision=True（👁 开关）→ 有图会话一律看图
        yield (文本增量, session_id, False)；最后一条 (完整回复, session_id, {"mode": ...})。"""
        def gen():
            with self.lock:
                session_id2, s = self.get_session(session_id)
                self._set_title(s, message)
                if image is not None:
                    s.image = image
                    s.has_image = True
                    # 新图：视觉上下文从当前消息尾部开始，旧图问答不进 VLM 上下文（防串图）
                    s.vision_start = len(s.messages)
                    s.messages.append({"role": "user", "content": "<|image|>"})

                use_vision = False
                if s.has_image:
                    # 确保视觉上下文以 <|image|> 占位开头（旧会话恢复/历史裁剪后可能缺失）
                    if (s.vision_start >= len(s.messages)
                            or s.messages[s.vision_start].get("content") != "<|image|>"):
                        s.vision_start = len(s.messages)
                        s.messages.append({"role": "user", "content": "<|image|>"})
                    use_vision = force_vision or self._looks_image_related(message)

                if use_vision:
                    mode = "vision"
                    # 看图会话不拼人设：实测人设文本会污染图像提问（踩坑 #12）
                    s.messages.append({"role": "user", "content": message})
                    tok = self.vlm_tok
                    prompt = tok.apply_chat_template(
                        s.messages[s.vision_start:],
                        add_generation_prompt=True, tokenize=False)
                    ids = tok.encode(prompt)
                    img_id = tok.convert_tokens_to_ids("<|image|>")
                    img_pos = ids.index(img_id) if img_id in ids else -1
                    img_feats = self._encode_image(s.image)
                    model = self.vlm_model
                else:
                    mode = "text"
                    msg = self._apply_persona(session_id2, s, message)
                    s.messages.append({"role": "user", "content": msg})
                    # 文本模型不认识 <|image|>：历史里剥掉占位符，空消息直接丢弃
                    hist = []
                    for m in s.messages:
                        content = (m["content"].replace("<|image|>\n", "")
                                   .replace("<|image|>", "").strip())
                        if content:
                            hist.append({"role": m["role"], "content": content})
                    tok = self.text_tok
                    prompt = tok.apply_chat_template(
                        hist, add_generation_prompt=True, tokenize=False)
                    ids = tok.encode(prompt)
                    img_feats, img_pos = None, -1
                    model = self.text_model

                parts = []
                for d in self._generate(model, tok, ids, img_feats, img_pos, **kw):
                    parts.append(d)
                    yield d, session_id2, False
                reply = "".join(parts)
                s.messages.append({"role": "assistant", "content": reply})
                # 历史裁剪：同步平移视觉上下文起点（占位被挤掉时 auto 补回）
                trimmed = len(s.messages) - self.max_history
                if trimmed > 0:
                    s.messages = s.messages[-self.max_history:]
                    s.vision_start = max(0, s.vision_start - trimmed)
                self._persist()
                yield reply, session_id2, {"mode": mode}
        return gen()

    def see_stream(self, session_id, question, image=None, **kw):
        """看图问答（流式生成器，强制看图）。
        旧接口保留（/api/vision/stream、/vision 用）；网页主入口走 auto_stream。
        yield (文本增量, session_id, False)；最后一条 (完整回复, session_id, {"mode": ...})。"""
        def gen():
            with self.lock:
                s = self.sessions.get(session_id)
                has = (image is not None) or (s is not None and s.has_image)
            if not has:
                raise ValueError("该会话还没有图片，先传 image")
            yield from self.auto_stream(session_id, question, image=image,
                                       force_vision=True, **kw)
        return gen()

    def health(self):
        info = {
            "ok": True,
            "device": self.device,
            "loaded_sec": round(self.loaded, 1),
            "sessions": len(self.sessions),
            "models": {
                "text": os.environ.get("NOVAMIND_TEXT_WEIGHT", "sft_clean_0.3b"),
                "vlm": self.vlm_tag,
                "name": "星眸 XingMou 0.3B",
                "tools": ["get_time", "create_file", "create_folder",
                          "read_file", "list_dir"] if self.enable_tools else [],
                "tool_model_loaded": self.tool_model is not None,
            },
        }
        if self.device == "cuda":
            info["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        return info


# --------------------------------------------------------------------------- #
#  自测入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(__doc__)
    print(f"model_store = {MODEL_STORE}")
    agent = NovaMindAgent()
    print(agent.health())

    print("\n--- 纯文字对话 ---")
    reply, sid = agent.chat("test", "你好，介绍一下你自己")
    print(f"[{sid}] {reply}")

    demo = os.path.join(_ROOT, "serve", "_demo.jpg")
    if os.path.exists(demo):
        print("\n--- 看图 ---")
        img = Image.open(demo).convert("RGB")
        reply, sid = agent.see("test-img", "这张图里有什么？", image=img)
        print(f"[{sid}] {reply}")
