"""
================================================================================
server.py —— 星眸 XingMou 本地服务（独立作品：网页 + API + OpenAI 兼容）
================================================================================

启动：
    uvicorn serve.server:app --host 127.0.0.1 --port 8787
    （或双击 serve/start_agent.bat）

网页：
    http://127.0.0.1:8787/     星眸对话页（独立前端，不依赖 pi/pi-web）

接口：
    GET  /health              状态
    GET  /api/sessions        会话列表
    DELETE /api/sessions/{id} 删除会话
    POST /api/chat/stream     纯文字对话（SSE 流式）
    POST /api/vision/stream   看图对话（SSE 流式）
    POST /chat /vision /clear 旧 JSON 接口（兼容保留）
    POST /v1/chat/completions OpenAI 兼容（第三方客户端用）
    GET  /v1/models           模型列表
================================================================================
"""

import base64
import json
import os
import re
import sys
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from serve.agent import NovaMindAgent

app = FastAPI(title="星眸 XingMou", version="2.0")
agent: NovaMindAgent = None
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# --------------------------------------------------------------------------- #
#  请求模型
# --------------------------------------------------------------------------- #
class ChatReq(BaseModel):
    session_id: str = ""
    message: str


class VisionReq(BaseModel):
    session_id: str = ""
    question: str
    image_path: str = ""
    image_b64: str = ""


class ClearReq(BaseModel):
    session_id: str = ""


@app.on_event("startup")
def _startup():
    global agent
    agent = NovaMindAgent()


@app.get("/health")
def health():
    return agent.health()


@app.post("/chat")
def chat(req: ChatReq):
    if not req.message.strip():
        return {"reply": "", "session_id": req.session_id or "default"}
    try:
        reply, sid = agent.chat(req.session_id, req.message)
        return {"reply": reply, "session_id": sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/vision")
def vision(req: VisionReq):
    try:
        image = agent.load_image(req.image_path or None, req.image_b64 or None)
        if image is None and not req.image_b64 and not req.image_path:
            # 允许不传图：沿用会话里已有的图继续追问
            image = None
        reply, sid = agent.see(req.session_id, req.question, image=image)
        return {"reply": reply, "session_id": sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/clear")
def clear(req: ClearReq):
    agent.clear_session(req.session_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
#  星眸网页专用 API
# --------------------------------------------------------------------------- #
class StreamChatReq(BaseModel):
    session_id: str = ""
    message: str
    image_b64: str = ""
    force_vision: bool = False
    temperature: float = None
    top_p: float = None
    top_k: int = None
    max_new_tokens: int = None


class StreamVisionReq(BaseModel):
    session_id: str = ""
    question: str
    image_b64: str = ""
    temperature: float = None
    top_p: float = None
    top_k: int = None
    max_new_tokens: int = None


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": agent.list_sessions()}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    agent.clear_session(sid)
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
def session_messages(sid: str):
    data = agent.get_messages(sid)
    if data is None:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    return data


def _gen_kw(req):
    kw = {}
    for k in ("temperature", "top_p", "top_k", "max_new_tokens"):
        v = getattr(req, k, None)
        if v is not None:
            kw[k] = v
    return kw


def _sse_chat_stream(req, image=None, force_vision=False, message=None):
    """统一 SSE 流：auto_stream 自动路由（带图强制看图，有图会话自动判断）。"""
    t0 = time.time()
    text = message if message is not None else req.message

    def gen():
        try:
            for delta, sid, done in agent.auto_stream(
                    req.session_id, text, image=image,
                    force_vision=force_vision, **_gen_kw(req)):
                if isinstance(done, dict):
                    speed = len(delta) / max(time.time() - t0, 1e-6)
                    s = agent.sessions.get(sid)
                    yield _sse({"type": "done", "reply": delta, "session_id": sid,
                                "speed": round(speed, 1),
                                "mode": done.get("mode", "text"),
                                "has_image": bool(s and s.has_image)})
                else:
                    yield _sse({"type": "delta", "text": delta})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat/stream")
def chat_stream(req: StreamChatReq):
    """网页统一入口（自动路由）：带图 → 看图；有图无图自动判断；force_vision → 强制看图。
    事件：delta / done（含 mode + has_image）/ error。"""
    image = None
    if req.image_b64:
        image = agent.load_image(image_b64=req.image_b64)
    return _sse_chat_stream(req, image=image, force_vision=req.force_vision)


@app.post("/api/vision/stream")
def vision_stream(req: StreamVisionReq):
    """看图对话，SSE 流式（强制看图，旧接口保留）。首次传 image_b64，后续追问可不传。"""
    image = None
    if req.image_b64:
        image = agent.load_image(image_b64=req.image_b64)
    return _sse_chat_stream(req, image=image, force_vision=True,
                            message=req.question)


# --------------------------------------------------------------------------- #
#  OpenAI 兼容（pi 自定义 provider 用）
# --------------------------------------------------------------------------- #
def _extract_images(content):
    """从 OpenAI 消息 content 里抽 data-uri 图片，返回 (纯文本, [PIL图])。"""
    imgs = []
    if isinstance(content, str):
        return content, imgs
    parts, text = [], []
    for c in content:
        if c.get("type") == "text":
            text.append(c.get("text", ""))
        elif c.get("type") == "image_url":
            url = c["image_url"].get("url", "")
            m = re.match(r"data:image/\w+;base64,(.+)", url, re.S)
            if m:
                imgs.append(agent.load_image(image_b64=m.group(1)))
    return "".join(text), imgs


def _sse(payload: dict):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(raw: dict):
    """OpenAI 兼容：纯文字走文本模型；带图走 VLM。支持 stream=true SSE。
    注意：必须是 sync def（FastAPI 会丢线程池），否则 GPU 生成会阻塞事件循环。"""
    req_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    model_name = raw.get("model", "xingmou-vl-0.3b")
    stream = raw.get("stream", False)
    msgs = raw.get("messages", [])
    # pi 默认发 max_completion_tokens（已配 compat.maxTokensField=max_tokens，双兼容）
    mt = raw.get("max_tokens") or raw.get("max_completion_tokens") or 256
    max_tokens = min(int(mt), 512)

    # 组装历史 + 找图
    # ⚠️ pi/pi-web 会发来整套编程智能体系统提示词（英文、超长、带工具说明），
    #    对 0.3B 中文聊天模型是纯噪声：直接丢弃 system/developer。
    history = []
    for m in msgs:
        role, content = m.get("role", "user"), m.get("content", "")
        text, imgs = _extract_images(content)
        if role in ("system", "developer"):
            continue
        if role == "assistant" and not text.strip():
            continue
        history.append({
            "role": "user" if role != "assistant" else "assistant",
            "content": text.strip(),
            "image": imgs[-1] if imgs else None,
        })

    # 小模型长历史必乱：只保留最近 6 条（3 个来回）
    history = history[-6:]

    # 图：仅最近 6 条里的最新一张生效；旧图自动过期（防止“问天气还在看图”的串台）
    image = None
    for m in reversed(history):
        if m["role"] == "user" and m["image"] is not None:
            image = m["image"]
            break

    # 人设永远放在最前（防长对话里被截掉后退回背死的自我介绍）；
    # 但看图请求不拼人设（人设词会污染图像提问，实测导致答非所问）
    if agent.persona and image is None:
        if history and history[0]["role"] == "user":
            history[0]["content"] = agent.persona + "\n\n" + history[0]["content"]
        else:
            history.insert(0, {"role": "user", "content": agent.persona, "image": None})

    # 最后一条 user 消息前加直接指令（对抗“问什么都回自我介绍”的罐头话术）
    # 看图请求不加（同上，避免污染图像提问）
    if image is None:
        steer = "（直接回答下面的问题，不要答非所问，用中文。）"
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                history[i]["content"] = steer + "\n" + history[i]["content"]
                break

    # 图占位符加回带图的那条消息
    for m in history:
        if m["image"] is not None:
            m["content"] = "<|image|>\n" + m["content"]

    if not history:
        history = [{"role": "user", "content": "你好", "image": None}]

    if image is not None:
        model, tok = agent.vlm_model, agent.vlm_tok
    else:
        model, tok = agent.text_model, agent.text_tok

    # 编码；超长时从第 2 条开始丢旧消息（人设第 1 条永远保留）
    while True:
        prompt = tok.apply_chat_template(
            [{k: v for k, v in m.items() if k != "image"} for m in history],
            add_generation_prompt=True, tokenize=False)
        ids = tok.encode(prompt)
        if len(ids) <= 1200 or len(history) <= 2:
            break
        history.pop(1)

    img_feats, img_pos = None, -1
    if image is not None:
        img_id = tok.convert_tokens_to_ids("<|image|>")
        if img_id in ids:
            img_pos = ids.index(img_id)
            img_feats = agent._encode_image(image)

    # 诊断日志（排查用）：记录每请求的角色/长度/是否带图
    try:
        log_line = (f"{time.strftime('%H:%M:%S')} msgs={len(msgs)} "
                    f"roles={[m.get('role') for m in msgs]} "
                    f"prompt_tokens={len(ids)} image={image is not None}\n")
        with open(os.path.join(os.path.dirname(__file__), "v1_log.txt"), "a",
                  encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        pass

    agent.max_new_tokens = max_tokens

    if not stream:
        with agent.lock:
            reply = "".join(agent._generate(model, tok, ids, img_feats, img_pos))
        return {
            "id": req_id, "object": "chat.completion", "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(reply),
                      "total_tokens": len(ids) + len(reply)},
        }

    def gen():
        # sync generator：Starlette 会在线程池里迭代，不阻塞事件循环
        n_gen = 0
        with agent.lock:
            yield _sse({"id": req_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model_name,
                        "choices": [{"index": 0, "delta": {"role": "assistant",
                                     "content": ""}, "finish_reason": None}]})
            for delta in agent._generate(model, tok, ids, img_feats, img_pos):
                n_gen += len(delta)
                yield _sse({"id": req_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": delta},
                                         "finish_reason": None}]})
            yield _sse({"id": req_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(ids),
                                  "completion_tokens": n_gen,
                                  "total_tokens": len(ids) + n_gen}})
            yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": "xingmou-vl-0.3b", "object": "model",
         "owned_by": "xingmou", "created": 0},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("NOVAMIND_PORT", "8787")))


# 静态网页（必须最后挂载，API 路由优先）
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
