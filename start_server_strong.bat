@echo off
REM 16GB 显存 — 单盘最强：1 个引擎占满 GPU，每手 1500 visits（更慢、更强）
REM 网页仍可用 ?katagoVisits=2000 进一步加码
set KATAGO_VRAM_GB=16
set KATAGO_ENGINE_POOL_SIZE=1
set KATAGO_DEFAULT_MODEL_BASENAME=kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz
set KATAGO_DEFAULT_VISITS=1500
set KATAGO_CFG=gtp_cuda.cfg
set HTTP_PORT=5001
cd /d "%~dp0"
echo Starting KataGo (max strength single game, visits=%KATAGO_DEFAULT_VISITS%)...
node server.js
pause
