#!/usr/bin/env python
# coding: utf-8
"""
Katago 棋盘识别后端：调用 go-board-detect（moku-v1 + OpenCV 网格）。

协议与 board_recognizer.py 相同：
  - stdin: 图片二进制
  - stdout: JSON { stones:[{x,y,color,conf}], boardSize, blackCount, whiteCount, debugSummary, ... }
  - BOARD_PREVIEW_ONLY=1 时额外返回 previewImageData
  - BOARD_MANUAL_QUAD: 可选 JSON 四角 [[x,y]×4]，作为木纹外框提示

环境变量：
  GO_BOARD_DETECT_DIR  go-board-detect 仓库根目录
  MOKU_MODEL_DIR       moku-v1 权重目录（含 model.safetensors）
  BOARD_GBD_CONF       检测置信度，默认 0.05
  BOARD_GBD_DEVICE     cuda:0 / cpu，默认自动
  BOARD_SIZE           9/13/19，默认 19
"""

from __future__ import annotations

import base64
import importlib.machinery
import json
import os
import sys
from pathlib import Path


def fail(msg: str) -> None:
    sys.stderr.write(str(msg))
    sys.exit(1)


def read_input_bytes() -> bytes:
    data = sys.stdin.buffer.read()
    if not data:
        fail("未接收到图片数据")
    return data


def _candidate_gbd_dirs() -> list[Path]:
    env = os.environ.get("GO_BOARD_DETECT_DIR", "").strip().strip('"').strip("'")
    out: list[Path] = []
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    out.extend(
        [
            here / "go-board-detect",
            Path(r"D:\AI-AGent-Learning\go-board-detect"),
            Path(r"D:\AI-AGent-Learning\11-Pytorch与视觉检测\yolo-cases"),
        ]
    )
    seen = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def resolve_gbd_dir() -> Path:
    for d in _candidate_gbd_dirs():
        if (d / "6-yolo-go-to-sgf.py").exists() and (d / "moku_infer.py").exists():
            return d
    fail(
        "找不到 go-board-detect（需要 6-yolo-go-to-sgf.py 与 moku_infer.py）。"
        "请设置环境变量 GO_BOARD_DETECT_DIR"
    )
    raise SystemExit(1)


def resolve_moku_dir(gbd_dir: Path) -> Path:
    env = os.environ.get("MOKU_MODEL_DIR", "").strip().strip('"').strip("'")
    cands = []
    if env:
        cands.append(Path(env))
    cands.extend(
        [
            gbd_dir / "models" / "moku-v1",
            Path(r"D:\AI-AGent-Learning\11-Pytorch与视觉检测\yolo-cases\models\moku-v1"),
            Path(r"D:\AI-AGent-Learning\go-board-detect\models\moku-v1"),
        ]
    )
    for d in cands:
        if (d / "model.safetensors").exists() and (d / "config.json").exists():
            return d
    # 交给 moku_infer 从 HuggingFace 下载到 gbd_dir/models/moku-v1
    return gbd_dir / "models" / "moku-v1"


def load_sgf_module(gbd_dir: Path):
    from importlib.util import module_from_spec, spec_from_loader

    path = gbd_dir / "6-yolo-go-to-sgf.py"
    loader = importlib.machinery.SourceFileLoader("yolo_go_to_sgf_gbd", str(path))
    spec = spec_from_loader(loader.name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def parse_manual_quad():
    raw = os.environ.get("BOARD_MANUAL_QUAD", "").strip()
    if not raw:
        return None
    try:
        import numpy as np

        pts = json.loads(raw)
        arr = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        return arr
    except Exception as exc:
        fail(f"BOARD_MANUAL_QUAD 无效: {exc}")
        return None


def downscale_bgr(img, max_edge: int = 1920):
    import cv2

    raw = os.environ.get("BOARD_RECOGNIZER_MAX_LONG_EDGE", str(max_edge)).strip()
    try:
        max_edge = int(raw)
    except ValueError:
        max_edge = 1920
    if max_edge <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return img
    scale = max_edge / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def build_preview_data_url(sgf_mod, img, intersections, grid, meta, board_size: int) -> str:
    import base64 as b64
    import cv2

    overlay = sgf_mod.draw_overlay(
        img, intersections, grid, board_size, quad=meta.get("quad") if meta else None
    )
    ok, png = cv2.imencode(".png", overlay)
    if not ok:
        fail("预览图生成失败")
    return "data:image/png;base64," + b64.b64encode(png.tobytes()).decode("ascii")


_detector = None
_sgf_mod = None
_gbd_dir = None


def get_runtime():
    global _detector, _sgf_mod, _gbd_dir
    if _detector is not None and _sgf_mod is not None:
        return _detector, _sgf_mod, _gbd_dir

    gbd_dir = resolve_gbd_dir()
    moku_dir = resolve_moku_dir(gbd_dir)
    sys.path.insert(0, str(gbd_dir))

    from moku_infer import MokuDetector

    conf = float(os.environ.get("BOARD_GBD_CONF", "0.05") or "0.05")
    device_env = os.environ.get("BOARD_GBD_DEVICE", "").strip()
    if not device_env:
        try:
            import torch

            device_env = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_env = "cpu"

    _detector = MokuDetector(model_dir=moku_dir, device=device_env, conf=conf)
    _sgf_mod = load_sgf_module(gbd_dir)
    _gbd_dir = gbd_dir
    return _detector, _sgf_mod, _gbd_dir


def detect_board_and_stones(image_bytes: bytes, preview_only: bool = False) -> dict:
    try:
        import cv2
        import numpy as np
    except Exception:
        fail("缺少依赖：请安装 opencv-python 和 numpy")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        fail("无法解析图片，请确认是有效的图片文件")
    img = downscale_bgr(img)

    detector, sgf_mod, gbd_dir = get_runtime()
    conf = float(os.environ.get("BOARD_GBD_CONF", "0.05") or "0.05")
    board_size = int(os.environ.get("BOARD_SIZE", "19") or "19")
    if board_size not in (9, 13, 19):
        board_size = 19

    wood_quad = parse_manual_quad()
    packed = sgf_mod.recognize_moku_bgr(
        detector,
        img,
        board_size=board_size,
        conf=conf,
        wood_quad=wood_quad,
        image_name="upload.jpg",
    )

    result = {
        "stones": packed["stones"],
        "boardSize": packed["boardSize"],
        "blackCount": packed["blackCount"],
        "whiteCount": packed["whiteCount"],
        "debugCandidates": [],
        "debugSummary": {
            **(packed.get("debugSummary") or {}),
            "gbdDir": str(gbd_dir),
        },
    }
    if packed.get("sgf"):
        result["sgf"] = packed["sgf"]

    if preview_only:
        intersections = packed.get("_intersections")
        grid = packed.get("_grid")
        meta = packed.get("_meta") or {}
        if intersections and grid is not None:
            result["previewImageData"] = build_preview_data_url(
                sgf_mod, img, intersections, grid, meta, board_size
            )
        else:
            fail("识别成功但无法生成预览图")

    return result


def main():
    image_bytes = read_input_bytes()
    preview_only = os.environ.get("BOARD_PREVIEW_ONLY", "").strip() == "1"
    # 识别过程中的 print / ultralytics 日志不得污染 stdout（Node 只解析 JSON）
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = detect_board_and_stones(image_bytes, preview_only=preview_only)
    finally:
        sys.stdout = real_stdout
    for k in list(result.keys()):
        if str(k).startswith("_"):
            result.pop(k, None)
    real_stdout.write(json.dumps(result, ensure_ascii=False))
    real_stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(str(exc))
        sys.exit(1)
