# 星眸 XingMou · 0.3B 多模态智能体

> 由日落星辉开发的独立本地作品：一个能聊天、能识图的 0.3B 小模型智能体，
> 自带完整网页界面，完全独立运行，不依赖 pi / pi-web 或其他框架。

> **【当前版本】纯聊天 + 识图版**：实测 DPO 版和工具决策循环反而让体验退化，已回退——
> 文字模型用 sft_clean_0.3b（DPO 前版本），工具循环停用（代码保留在 agent.py，前端开关已移除），
> 识图保持 vlm_s2v2_0.3b + SigLIP2 不变。

## 快速开始

```bat
双击 serve/start_agent.bat
```

浏览器打开 **http://127.0.0.1:8787** 即可使用。

## 产品组成

```
serve/
├── agent.py                 # 智能体核心：双模型路由 + 多会话 + 流式生成 + 人设
├── server.py                # FastAPI：网页 + API + OpenAI 兼容接口
├── start_agent.bat          # 一键启动（conda env novamind + 4060 GPU）
├── web/                     # 独立网页前端（纯原生 HTML/CSS/JS，零依赖）
│   ├── index.html           #   深空星夜主题界面
│   ├── style.css            #   响应式样式
│   └── app.js               #   流式渲染 + 会话管理 + 图片上传
└── model_store/             # 权重（从服务器下载，共 2.8G：双套视觉塔）
    ├── out/                 #   sft_clean_0.3b（文字）+ vlm_s2v2_0.3b（识图）+ 投影层
    ├── tokenizer/ vlm_tokenizer/
    └── siglip2-256/         # SigLIP2 完整目录（1.5G，vision-only 会权重错位，勿精简）
```

## 网页功能

- 💬 流式对话（星夜界面，打字机光标，逐字输出 + 速度统计）
- 🖼 识图：上传/拖拽/粘贴图片，多轮追问
- 📁 会话管理：自动标题、历史列表、切换/删除
- ⚙ 生成参数：温度 / top_p / top_k / 最大长度实时调节
- 📊 状态栏：GPU 就绪状态、显存占用
- 📱 响应式布局，手机也能用

## 模型路由（内部，2026-09-13 自动路由版）

| 输入 | 路由 |
|---|---|
| 纯文字会话 | sft_clean_0.3b（中文专精，身份星眸 + 人设） |
| 带新图的消息 | vlm_s2v2_0.3b + SigLIP2（图放视觉上下文开头，不加人设） |
| 有图会话的后续问题 | 自动判断：图相关词/代词/短追问 → 看图；其余 → 纯文字（含完整记忆） |
| 👁 开关开启 | 强制看图（有图会话所有问题带图回答） |

发新图不清空会话记忆（文字历史全保留，旧图问答不进新图的视觉上下文）。

## API

| 接口 | 说明 |
|---|---|
| `GET /health` | 状态（device / 显存 / 会话数） |
| `GET /api/sessions` | 会话列表（id / 标题 / 是否看图） |
| `GET /api/sessions/{id}/messages` | 会话历史 |
| `DELETE /api/sessions/{id}` | 删除会话 |
| `POST /api/chat/stream` | 统一入口（自动路由：message + 可选 image_b64 + force_vision） |
| `POST /api/vision/stream` | 看图对话（强制看图，旧接口保留） |
| `POST /v1/chat/completions` | OpenAI 兼容（第三方客户端可选接入） |

SSE 事件：`delta`（增量文本）→ `done`（完整回复 + session_id + 速度 + 模式）→
`error`（异常）。

## 能力边界

- 0.3B 参数量：聊天“能用”（常识/算术/闲聊 OK，复杂推理不行）
- 识图“说个大概”：主体/场景/颜色可以，大字招牌能读，小字精细 OCR 不行
- 回答偶尔会跑偏/幻觉，这是小模型的天花板，不是 bug

## 踩坑记录（本地部署）

1. **看图会话不能拼人设**：人设文本（含"星眸/XingMou"）会污染图像提问，
   小模型把注意力放在人设词上，答非所问（"没看到XingMou签名"）。
   解法：看图会话直接问图，只有纯文字会话拼人设。
2. **SigLIP2 不能存 vision-only 目录**：vm.half().save_pretrained() 再加载会权重错位
   （服务器上 conv2d 报错，本地输出垃圾特征）。解法：直接用完整 siglip2-256 目录，
   NovaMindVision 自动取 .vision_model。SigLIP1 的 vision-only 目录正常。
3. **.bat 必须 CRLF**：LF 换行的 bat 在 cmd 下静默不执行。

**A/B 切回旧版**：环境变量 `NOVAMIND_VLM_WEIGHT=vlm_s2_0.3b`
`NOVAMIND_VLM_PROJECTOR=vlm_projector_s2` `NOVAMIND_VISION=siglip256-vision-only`。

## 运行环境与配置

- 运行环境：conda 环境 `novamind`（Python 3.12 + torch 2.6+cu124），权重放在 `model_store/`
- 启动日志：`novamind_server.log`（排查启动问题先看这里）
- 文字模型切换：环境变量 `NOVAMIND_TEXT_WEIGHT`（默认 sft_clean_0.3b）
- 识图模型切换：环境变量 `NOVAMIND_VLM_WEIGHT` / `NOVAMIND_VLM_PROJECTOR` / `NOVAMIND_VISION`
- 人设修改：环境变量 `NOVAMIND_PERSONA`（置空关闭人设）
- A/B 切回旧版视觉：`NOVAMIND_VLM_WEIGHT=vlm_s2_0.3b`
  `NOVAMIND_VLM_PROJECTOR=vlm_projector_s2` `NOVAMIND_VISION=siglip256-vision-only`
