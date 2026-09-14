# -*- coding: utf-8 -*-
"""星眸 GitHub 演示视频录制脚本（Playwright 驱动真实界面）"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
IMG = r"C:\Users\44710\anime-avatar.png"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_raw")


def wait_done(page, timeout_ms=120000):
    """等待最新 assistant 消息出现 meta（done 事件已到达）。"""
    page.wait_for_function(
        """() => {
            const msgs = [...document.querySelectorAll('.msg.assistant')];
            if (!msgs.length) return false;
            const meta = msgs[msgs.length - 1].querySelector('.meta');
            return meta && meta.textContent.trim().length > 0;
        }""",
        timeout=timeout_ms,
    )
    page.wait_for_timeout(900)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    def log(tag):
        print(f"[{time.perf_counter()-t0:6.2f}s] {tag}", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=OUT_DIR,
            record_video_size={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        page.set_default_timeout(150000)
        page.goto(BASE, wait_until="networkidle")

        # 等待欢迎页 hero 渲染
        page.wait_for_selector(".hero-halo", timeout=30000)
        page.wait_for_timeout(1600)
        log("欢迎页展示完")

        # 采样参数调优（实测多轮验证：低温下回复稳定、不触发罐头）：
        #   temp 0.5 / top_p 0.85 / top_k 40 / 最大 140 token
        page.evaluate(
            """() => {
                const set = (id, v) => {
                    const el = document.getElementById(id);
                    el.value = v; el.dispatchEvent(new Event('input'));
                };
                set('setTemp', 0.5); set('setTopP', 0.85);
                set('setTopK', 40); set('setMaxLen', 140);
            }"""
        )

        # ---- 场景 A：纯文字聊天 ----
        print("[1/3] 文字聊天 ...", flush=True)
        page.fill("#input", "水的化学式是什么？")
        page.click("#btnSend")
        log("A 发送")
        page.wait_for_timeout(1400)          # 展示思考动画
        wait_done(page)
        log("A 完成")

        # ---- 场景 B：上传图片识图 ----
        print("[2/3] 识图 ...", flush=True)
        page.set_input_files("#fileInput", IMG)
        page.wait_for_timeout(800)           # 展示图片预览
        page.fill("#input", "用一句话描述这张图")
        page.click("#btnSend")
        log("B 发送")
        page.wait_for_timeout(1400)
        wait_done(page)
        log("B 完成")

        # ---- 场景 C：有图会话里问一般问题 → 自动路由到文字 ----
        print("[3/3] 自动路由 ...", flush=True)
        page.fill("#input", "你好")
        page.click("#btnSend")
        log("C 发送")
        page.wait_for_timeout(1400)
        wait_done(page)
        log("C 完成")

        page.wait_for_timeout(1200)
        ctx.close()
        browser.close()
    print("录制完成", flush=True)


if __name__ == "__main__":
    main()
