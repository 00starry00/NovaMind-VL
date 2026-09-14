@echo off
rem ============================================================
rem  NovaMind-VL 本地推理服务启动脚本
rem  依赖: conda 环境 novamind (CUDA torch)
rem  服务: http://127.0.0.1:8787  (健康检查 /health)
rem ============================================================
set PYTHONNOUSERSITE=1
cd /d C:\Users\44710\pi-cwd-20260828\NovaMind-VL
D:\Anaconda3\envs\novamind\python.exe -m uvicorn serve.server:app --host 127.0.0.1 --port 8787
