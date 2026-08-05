import json
import os
import sys

# Roboflow Universe「go-positions」托管模型；与 server.js 中 ROBOFLOW_MODEL_ENDPOINT_DEFAULT 保持一致。
DEFAULT_BOARD_ROBOFLOW_GO_POSITIONS_MODEL = "synthetic-data-3ol2y/go-positions/model/6"


def fail(msg: str):
    raise RuntimeError(msg)


def _roboflow_api_key_set():
    """子进程内是否可见 ROBOFLOW_API_KEY（仅布尔，不回传密钥内容）。"""
    return bool(os.environ.get("ROBOFLOW_API_KEY", "").strip())


# 最近一次 Roboflow 失败原因（供 debugSummary 使用，不含密钥）
_ROBOFLOW_LAST_ERROR = ""


def _roboflow_set_last_error(msg: str):
    global _ROBOFLOW_LAST_ERROR
    _ROBOFLOW_LAST_ERROR = (msg or "")[:900]


def _sanitize_env_wrapped_string(value):
    """
    去掉环境变量里常见的成对引号（Windows / JSON 复制粘贴），
    避免出现 URL 路径含 %22 导致 detect / serverless 请求失败。
    """
    s = (value or "").strip()
    for _ in range(5):
        if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
            s = s[1:-1].strip()
        else:
            break
    return s.strip()


def _roboflow_detect_inference_path_candidates(primary_endpoint: str):
    """
    Roboflow 官方 Python SDK（detect API）使用的路径为:
      POST https://detect.roboflow.com/{project_slug}/{version}
    不含 workspace。Universe 上复制的
      synthetic-data-3ol2y/go-positions/model/6
    必须解析为 go-positions/6；若直接请求
      serverless.../synthetic-data-.../go-positions/6 且带 multipart，会返回 405。

    可选环境变量 BOARD_ROBOFLOW_DETECT_PATH（如 go-positions/6）可覆盖自动解析。
    """
    override = _sanitize_env_wrapped_string(os.environ.get("BOARD_ROBOFLOW_DETECT_PATH", "")).strip().strip("/")
    if override:
        return [override]
    p = _sanitize_env_wrapped_string(primary_endpoint).strip().strip("/")
    if not p:
        return []
    if "/model/" in p:
        i = p.rfind("/model/")
        p = p[:i] + "/" + p[i + len("/model/") :]
    parts = [x for x in p.split("/") if x]
    if len(parts) < 2 or not parts[-1].isdigit():
        return []
    proj, ver = parts[-2], parts[-1]
    return [f"{proj}/{ver}"]


def _roboflow_inference_hosts():
    """
    Universe 文档（go-positions model/6）推荐:
      https://serverless.roboflow.com/{project}/{version}
    与 detect.roboflow.com 等价路径、请求体格式一致；未配置时先 detect 再 serverless（通常总等待更短）。
    BOARD_ROBOFLOW_INFERENCE_HOST 可设为单个 URL，或逗号分隔多个（按顺序尝试）。
    """
    raw = _sanitize_env_wrapped_string(os.environ.get("BOARD_ROBOFLOW_INFERENCE_HOST", "")).strip()
    if raw:
        parts = [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]
        return parts if parts else ["https://serverless.roboflow.com"]
    # 默认先 detect：与 serverless 等价路径时往往排队更短，避免先卡 serverless 再等 detect
    return ["https://detect.roboflow.com", "https://serverless.roboflow.com"]


def _board_warp_dst_size() -> int:
    """
    透视拉正后的棋盘边长（像素）。默认 1280（此前固定 1024），手机拍照压缩后
    更大分辨率能保留棋子边缘，显著降低漏检；可用环境变量 BOARD_WARP_DST_SIZE 覆盖。
    """
    try:
        v = int(os.environ.get("BOARD_WARP_DST_SIZE", "1280").strip())
    except Exception:
        v = 1280
    return max(896, min(1600, int(v)))


def _roboflow_limit_infer_bgr(img_bgr, max_side: int):
    """原图推理前可选缩小长边，减轻上传体积；max_side<=0 不缩放（与网页直接上传大图一致）。"""
    if max_side <= 0 or img_bgr is None:
        return img_bgr
    try:
        import cv2  # type: ignore

        h, w = img_bgr.shape[:2]
        m = max(int(h), int(w))
        if m <= max_side:
            return img_bgr
        scale = float(max_side) / float(m)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        return img_bgr


def _roboflow_traditional_fallback_pack(
    best_result,
    warped,
    wgray,
    xs,
    ys,
    model_endpoint,
    *,
    mode_tag: str,
    inference_source: str,
    extra_summary=None,
):
    """
    Roboflow 未产出可用检测时，仍返回传统分类结果（非 None），并在 debugSummary 标明原因。
    """
    stones_tr = best_result.get("stones") or []
    if not isinstance(stones_tr, list) or len(stones_tr) == 0:
        return None
    # best_result["stones"] 已在 _evaluate_quad_candidate 末尾做过 _apply_stone_post_filters；
    # 此处再跑一遍空间/白子反光启发式会二次误删（手机图常见），Roboflow 403 回退时漏检加剧。
    if os.environ.get("BOARD_FALLBACK_REAPPLY_POST_FILTER", "0").strip() == "1":
        stones_fb = _apply_stone_post_filters(warped, wgray, xs, ys, list(stones_tr))
    else:
        stones_fb = list(stones_tr)
    black_count = sum(1 for s in stones_fb if s.get("color") == "B")
    white_count = sum(1 for s in stones_fb if s.get("color") == "W")
    ds = dict(best_result.get("_debugSummary") or {})
    ds.update(
        {
            "mode": mode_tag,
            "modelEndpoint": model_endpoint,
            "roboflowKeyPresent": True,
            "inferenceSource": inference_source,
            "roboflowError": _ROBOFLOW_LAST_ERROR,
        }
    )
    if extra_summary:
        ds.update(extra_summary)
    return {
        "stones": stones_fb,
        "boardSize": 19,
        "blackCount": int(black_count),
        "whiteCount": int(white_count),
        "debugCandidates": best_result.get("_debugCandidates", []),
        "debugSummary": ds,
    }


def read_input_bytes() -> bytes:
    data = sys.stdin.buffer.read()
    if not data:
        fail("未接收到图片数据")
    return data


def _order_quad_points(pts):
    # 输入 4 个点，按 tl,tr,br,bl 顺序输出
    import numpy as np  # type: ignore

    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1]).reshape(-1)
    tl = pts[s.argmin()]
    br = pts[s.argmax()]
    tr = pts[d.argmax()]
    bl = pts[d.argmin()]
    return np.array([tl, tr, br, bl], dtype="float32")


def _rotate_image_keep_bounds(gray, angle_deg):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    h, w = gray.shape[:2]
    cx = w * 0.5
    cy = h * 0.5
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_v = abs(m[0, 0])
    sin_v = abs(m[0, 1])
    nw = int(round(h * sin_v + w * cos_v))
    nh = int(round(h * cos_v + w * sin_v))
    m[0, 2] += (nw * 0.5) - cx
    m[1, 2] += (nh * 0.5) - cy
    rot = cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inv = cv2.invertAffineTransform(m)
    return rot, m, inv


def _map_quad_affine(quad, inv_affine):
    import numpy as np  # type: ignore

    q = np.array(quad, dtype=np.float32).reshape(4, 2)
    ones = np.ones((4, 1), dtype=np.float32)
    qh = np.concatenate([q, ones], axis=1)  # [x, y, 1]
    mapped = (qh @ inv_affine.T).astype(np.float32)
    return mapped


def _clahe_gray(gray):
    import cv2  # type: ignore

    # 参考开源 gbr 思路：CLAHE 能缓解手机拍照时的局部过曝/阴影
    clahe = cv2.createCLAHE(clipLimit=2.75, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_board_quad(gray):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    h, w = gray.shape[:2]
    if min(h, w) > 1500:
        scale = 1200.0 / float(max(h, w))
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        upscale = 1.0 / scale
    else:
        small = gray
        upscale = 1.0

    hs, ws = small.shape[:2]

    def score_gridness(quad):
        # 将候选四边形拉正，按“像 19x19 网格”的程度打分
        dst_size = 420
        dst = np.array(
            [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
            dtype=np.float32,
        )
        oq = _order_quad_points(quad.astype(np.float32))
        m = cv2.getPerspectiveTransform(oq, dst)
        warp = cv2.warpPerspective(small, m, (dst_size, dst_size))
        wb = cv2.GaussianBlur(warp, (5, 5), 0)

        gx = cv2.Sobel(wb, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(wb, cv2.CV_32F, 0, 1, ksize=3)
        sx = np.mean(np.abs(gx), axis=0).astype(np.float32)
        sy = np.mean(np.abs(gy), axis=1).astype(np.float32)
        sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 17), 0).reshape(-1)
        sy = cv2.GaussianBlur(sy.reshape(-1, 1), (17, 1), 0).reshape(-1)

        xs = _find_19_grid_positions(sx)
        ys = _find_19_grid_positions(sy)
        if xs is None or ys is None:
            return -1.0

        dx = np.diff(np.array(xs, dtype=np.float32))
        dy = np.diff(np.array(ys, dtype=np.float32))
        if dx.size < 3 or dy.size < 3:
            return -1.0
        # 间距一致性越高越像棋盘
        reg_x = float(np.std(dx) / (np.mean(dx) + 1e-6))
        reg_y = float(np.std(dy) / (np.mean(dy) + 1e-6))
        spacing_score = 1.0 / (0.12 + reg_x + reg_y)

        line_score = float(np.mean(sx[xs]) + np.mean(sy[ys]))
        return line_score * spacing_score

    # 多路预处理，提升复杂背景/小棋盘检出率
    prep_images = []
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    prep_images.append(cv2.Canny(blur, 40, 120))
    prep_images.append(cv2.Canny(blur, 60, 180))

    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    prep_images.append(thr)

    candidates = []
    min_area = (hs * ws) * 0.012  # 允许较小棋盘
    for img_bin in prep_images:
        work = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(work, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri < 40:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2)
            area = cv2.contourArea(quad)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(quad)
            if cw < 20 or ch < 20:
                continue
            ar = cw / float(ch)
            if ar < 0.55 or ar > 1.85:
                continue
            rect_area = float(cw * ch)
            fill_ratio = float(area / (rect_area + 1e-6))
            if fill_ratio < 0.50:
                continue
            candidates.append(quad.astype(np.float32))

    if not candidates:
        return None

    # 去重（按中心和面积粗聚合）
    dedup = []
    for q in candidates:
        x, y, cw, ch = cv2.boundingRect(q.astype(np.int32))
        cx = x + cw * 0.5
        cy = y + ch * 0.5
        ar = cw / float(ch + 1e-6)
        area = float(cw * ch)
        keep = True
        for d in dedup:
            if abs(cx - d["cx"]) < 18 and abs(cy - d["cy"]) < 18 and abs(area - d["area"]) / max(area, d["area"]) < 0.18:
                keep = False
                break
        if keep:
            dedup.append({"quad": q, "cx": cx, "cy": cy, "ar": ar, "area": area})

    best_quad = None
    best_score = -1.0
    # 先按面积取前若干，再按网格性打分
    dedup.sort(key=lambda x: x["area"], reverse=True)
    for item in dedup[: min(32, len(dedup))]:
        q = item["quad"]
        gscore = score_gridness(q)
        if gscore < 0:
            continue
        # 轻度偏好面积大/长宽接近 1，但不偏好中心位置
        aspect_bonus = 1.0 / (0.35 + abs(item["ar"] - 1.0))
        area_bonus = (item["area"] / (hs * ws)) ** 0.18
        final_score = gscore * (0.75 + 0.25 * aspect_bonus) * area_bonus
        if final_score > best_score:
            best_score = final_score
            best_quad = q

    if best_quad is None:
        return None

    # 映射回原图坐标
    return (best_quad * upscale).astype(np.float32)


def _detect_board_quad_hough(gray):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    h, w = gray.shape[:2]
    if min(h, w) > 1700:
        scale = 1300.0 / float(max(h, w))
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        upscale = 1.0 / scale
    else:
        small = gray
        upscale = 1.0

    hs, ws = small.shape[:2]
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=55,
        minLineLength=int(max(hs, ws) * 0.10),
        maxLineGap=12,
    )
    if lines is None or len(lines) < 20:
        return None

    sx = np.zeros(ws, dtype=np.float32)
    sy = np.zeros(hs, dtype=np.float32)

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < 8:
            continue
        adx = abs(dx)
        ady = abs(dy)
        # 近竖线：给 x 方向投票
        if ady > adx * 1.35:
            xm = int(round((x1 + x2) * 0.5))
            if 0 <= xm < ws:
                sx[xm] += length
        # 近横线：给 y 方向投票
        if adx > ady * 1.35:
            ym = int(round((y1 + y2) * 0.5))
            if 0 <= ym < hs:
                sy[ym] += length

    if float(np.max(sx)) <= 0.0 or float(np.max(sy)) <= 0.0:
        return None

    sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 19), 0).reshape(-1)
    sy = cv2.GaussianBlur(sy.reshape(-1, 1), (19, 1), 0).reshape(-1)
    xs = _find_19_grid_positions(sx)
    ys = _find_19_grid_positions(sy)
    if xs is None or ys is None:
        return None

    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])
    if x1 - x0 < ws * 0.20 or y1 - y0 < hs * 0.20:
        return None

    quad_small = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )
    return (quad_small * upscale).astype(np.float32)


def _detect_board_quad_window_scan(gray):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    h, w = gray.shape[:2]
    long_side = max(h, w)
    scale = 1.0
    if long_side > 1100:
        scale = 900.0 / float(long_side)
    elif long_side < 500:
        scale = 500.0 / float(long_side)

    if abs(scale - 1.0) > 1e-3:
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        upscale = 1.0 / scale
    else:
        small = gray
        upscale = 1.0

    hs, ws = small.shape[:2]
    if min(hs, ws) < 220:
        return None

    # 快速打分：窗口是否包含规则网格（19 线间距接近等分）
    def grid_score(crop):
        if crop.size == 0:
            return -1.0
        norm = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(norm, (5, 5), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        sx = np.mean(np.abs(gx), axis=0).astype(np.float32)
        sy = np.mean(np.abs(gy), axis=1).astype(np.float32)
        sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 15), 0).reshape(-1)
        sy = cv2.GaussianBlur(sy.reshape(-1, 1), (15, 1), 0).reshape(-1)
        xs = _find_19_grid_positions(sx)
        ys = _find_19_grid_positions(sy)
        if xs is None or ys is None:
            return -1.0
        dx = np.diff(np.array(xs, dtype=np.float32))
        dy = np.diff(np.array(ys, dtype=np.float32))
        if dx.size < 5 or dy.size < 5:
            return -1.0
        reg_x = float(np.std(dx) / (np.mean(dx) + 1e-6))
        reg_y = float(np.std(dy) / (np.mean(dy) + 1e-6))
        if reg_x > 0.36 or reg_y > 0.36:
            return -1.0
        return float((np.mean(sx[xs]) + np.mean(sy[ys])) / (0.12 + reg_x + reg_y))

    # 粗到细窗口扫描：适配“棋盘小/偏角落/背景复杂”
    best = None
    best_score = -1.0
    min_side = int(min(hs, ws) * 0.22)
    max_side = int(min(hs, ws) * 0.92)
    if max_side <= min_side:
        return None

    for ratio in (0.92, 0.82, 0.72, 0.62, 0.52, 0.42, 0.32, 0.25):
        side = int(min(hs, ws) * ratio)
        if side < min_side or side > max_side:
            continue
        stride = max(12, side // 7)
        for y0 in range(0, hs - side + 1, stride):
            for x0 in range(0, ws - side + 1, stride):
                crop = small[y0 : y0 + side, x0 : x0 + side]
                s = grid_score(crop)
                if s <= 0:
                    continue
                area_bonus = (side / float(min(hs, ws))) ** 0.22
                final = s * (0.82 + 0.18 * area_bonus)
                if final > best_score:
                    best_score = final
                    best = (x0, y0, side)

    if best is None:
        return None

    x0, y0, side = best
    quad_small = np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float32,
    )
    return (quad_small * upscale).astype(np.float32)


def _find_19_grid_positions(score):
    import numpy as np  # type: ignore

    n = len(score)
    if n < 100:
        return None

    # 局部峰值候选
    cand = []
    for i in range(2, n - 2):
        v = score[i]
        if v >= score[i - 1] and v >= score[i + 1] and v >= score[i - 2] and v >= score[i + 2]:
            cand.append((float(v), i))
    if len(cand) < 10:
        return None
    cand.sort(reverse=True)
    cand_idx = sorted([i for _, i in cand[: min(140, len(cand))]])

    # 在候选中搜索“首尾线”，使 19 个等间距采样点总分最高
    best = None
    best_score = -1e18
    for ai in range(len(cand_idx)):
        a = cand_idx[ai]
        for bi in range(ai + 1, len(cand_idx)):
            b = cand_idx[bi]
            span = b - a
            if span < n * 0.42 or span > n * 0.95:
                continue
            step = span / 18.0
            if step < 8 or step > 70:
                continue
            total = 0.0
            valid = True
            pts = []
            for k in range(19):
                x = a + k * step
                xi = int(round(x))
                if xi < 0 or xi >= n:
                    valid = False
                    break
                pts.append(xi)
                total += float(score[xi])
            if valid and total > best_score:
                best_score = total
                best = pts

    if best is None:
        return None

    # 对每个线位做局部微调
    refined = []
    for x in best:
        l = max(0, x - 4)
        r = min(n - 1, x + 4)
        local = int(np.argmax(score[l : r + 1])) + l
        refined.append(local)

    # 单调递增和最小间距修正
    out = [refined[0]]
    for i in range(1, 19):
        v = refined[i]
        if v <= out[-1]:
            v = out[-1] + 1
        out.append(v)
    return out


def _sample_1d_linear(arr, x):
    import math

    n = len(arr)
    if n <= 0:
        return 0.0
    if x <= 0:
        return float(arr[0])
    if x >= n - 1:
        return float(arr[n - 1])
    i0 = int(math.floor(x))
    i1 = i0 + 1
    t = x - i0
    return float(arr[i0] * (1.0 - t) + arr[i1] * t)


def _find_regular_grid_positions_by_margin(score, board_span=19):
    """
    规则网格拟合：假设 19 条线等间距，只搜索“首线边距 m”。
    适合手动四角拉正后的棋盘（即使峰值受噪声干扰，等间距约束仍然稳）。
    """
    import numpy as np  # type: ignore

    n = len(score)
    if n < 120:
        return None

    best = None
    best_score = -1e18
    lines = board_span
    # 边距搜索范围：避免贴边和过于靠内
    m_min = max(8, int(round(n * 0.04)))
    m_max = int(round(n * 0.26))
    if m_max <= m_min:
        return None

    for m in range(m_min, m_max + 1):
        span = n - 1 - 2 * m
        if span <= 0:
            continue
        step = span / float(lines - 1)
        if step < 10 or step > 90:
            continue
        pts = [m + k * step for k in range(lines)]
        vals = np.array([_sample_1d_linear(score, p) for p in pts], dtype=np.float32)
        # 越多线条落在高响应处越好，同时偏好“响应方差小”（19线都被稳定命中）
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        fit_score = mean_v - 0.22 * std_v
        if fit_score > best_score:
            best_score = fit_score
            best = [int(round(p)) for p in pts]

    if best is None:
        return None
    # 单调修正
    out = [max(0, min(n - 1, best[0]))]
    for i in range(1, len(best)):
        v = max(0, min(n - 1, best[i]))
        if v <= out[-1]:
            v = min(n - 1, out[-1] + 1)
        out.append(v)
    return out


def _refine_grid_equal_spacing_1d(score, xs_hint):
    """
    在 1D 线响应曲线上，用「首线位置 m + 等间距 step」联合搜索，使 19 条线处的采样得分和最大。
    解决峰值法在噪声/阴影下整体偏一格或间距微变导致的「交叉点与真实网格错位」。
    """
    import numpy as np  # type: ignore

    n = len(score)
    if n < 120 or xs_hint is None or len(xs_hint) != 19:
        return None

    arr = np.array(score, dtype=np.float32)
    m0 = float(np.clip(xs_hint[0], 4.0, n - 5.0))
    diffs = np.diff(np.array(xs_hint, dtype=np.float32))
    s0 = float(np.median(diffs))
    s0 = float(np.clip(s0, 9.0, min(88.0, (n - 1 - 2 * max(6.0, m0 * 0.5)) / 18.0 * 1.05)))

    best_sum = -1e18
    best_m, best_s = m0, s0
    # 在粗定位附近细搜 m、step，保证 m + 18*step 落在图像内
    for m in np.linspace(max(3.0, m0 - 18.0), min(n - 4.0, m0 + 18.0), 35):
        for s in np.linspace(s0 * 0.88, s0 * 1.12, 33):
            last = m + 18.0 * s
            if last >= n - 2.0 or m < 2.0:
                continue
            total = 0.0
            ok = True
            for k in range(19):
                p = m + k * s
                if p < 0.5 or p >= n - 1.5:
                    ok = False
                    break
                total += float(_sample_1d_linear(arr, p))
            if ok and total > best_sum:
                best_sum = total
                best_m, best_s = m, s

    out = []
    for k in range(19):
        out.append(int(round(best_m + k * best_s)))
    out[0] = max(0, min(n - 1, out[0]))
    for i in range(1, 19):
        out[i] = max(out[i - 1] + 1, min(n - 1, out[i]))
    return out


def _refine_grid_global_shift_1d(score, xs):
    """
    在「整格」尺度上尝试将 19 条线整体平移 -1/0/+1 格（步长约 median(diff)）。
    Sobel 对线段的响应峰常偏向线的一侧，等间距拟合可能使整条网格相对真实交叉点偏一格，
    表现为棋盘上子力整体偏左或偏右（列 gx）/ 偏上或偏下（行 gy）。
    在若干平移候选中选 sum(score[x]) 最大者。
    """
    import numpy as np  # type: ignore

    n = len(score)
    if len(xs) != 19:
        return xs
    arr = np.array(xs, dtype=np.float32)
    step = float(np.median(np.diff(arr)))
    step = float(np.clip(step, 7.0, 95.0))

    def _apply_shift(k_int):
        shifted = [int(round(float(arr[i]) + k_int * step)) for i in range(19)]
        shifted[0] = max(0, min(n - 1, shifted[0]))
        for i in range(1, 19):
            # 防止因单调约束 + clamp 次序导致出现 n（如 n=1024，出现 index 1024）。
            # 当前一条已经贴到边界时，后续直接饱和到 n-1，避免越界。
            lo = shifted[i - 1] + 1
            if lo > n - 1:
                shifted[i] = n - 1
            else:
                shifted[i] = max(lo, min(n - 1, shifted[i]))
        return shifted

    best = None
    best_sum = -1e18
    best_tie = -999
    for k in (-2, -1, 0, 1, 2):
        shifted = _apply_shift(k)
        total = sum(float(score[x]) for x in shifted)
        tie = -abs(k)  # 总分接近时优先不平移，减少误修正
        if total > best_sum + 1e-5 or (abs(total - best_sum) <= 1e-5 and tie > best_tie):
            best_sum = total
            best_tie = tie
            best = shifted
    return best


def _pick_best_grid_positions(score):
    """
    组合两种网格定位：
    - 峰值法（适合线条清晰）
    - 规则等间距拟合法（适合手动四角后的噪声/轻微畸变）
    """
    import numpy as np  # type: ignore

    cand_a = _find_19_grid_positions(score)
    cand_b = _find_regular_grid_positions_by_margin(score, board_span=19)
    candidates = [c for c in [cand_a, cand_b] if c is not None]
    if not candidates:
        return None

    best = None
    best_score = -1e18
    for xs in candidates:
        arr = np.array(xs, dtype=np.int32)
        vals = np.array([float(score[i]) for i in arr], dtype=np.float32)
        d = np.diff(arr.astype(np.float32))
        reg = float(np.std(d) / (np.mean(d) + 1e-6))
        s = float(np.mean(vals)) / (0.10 + reg)
        if s > best_score:
            best_score = s
            best = xs
    return best


def _patch_gradient_anisotropy(wgray, cx, cy, radius):
    """
    局部 |∂x| 与 |∂y| 平均幅值之比：网格线反光多为沿某一方向的细长亮条（各向异性大），
    真白子近似圆形（各向异性接近 1）。
    """
    import numpy as np  # type: ignore

    h, w = wgray.shape[:2]
    cx = int(cx)
    cy = int(cy)
    r = max(3, int(radius))
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    patch = wgray[y0:y1, x0:x1].astype(np.float32)
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 1.0
    dx = np.abs(np.diff(patch, axis=1))
    dy = np.abs(np.diff(patch, axis=0))
    mdx = float(np.mean(dx))
    mdy = float(np.mean(dy))
    return max(mdx, mdy) / (min(mdx, mdy) + 1e-3)


def _filter_stones_spatial_board_face(wgray, xs, ys, stones):
    """
    去掉「真实在棋盘 19 路网格外」却被吸附到边缘格点的子（常见于边界外的提子、盒盖等）。
    用局部亮/暗团块的加权质心与交叉点 (xs[gx],ys[gy]) 的距离，以及质心是否落在
    首末条线围成的盘面矩形内（带容差）来判断。
    """
    import numpy as np  # type: ignore

    if len(xs) != 19 or len(ys) != 19 or not stones:
        return stones

    arr_x = np.array(xs, dtype=np.float32)
    arr_y = np.array(ys, dtype=np.float32)
    step_x = float(np.median(np.diff(arr_x)))
    step_y = float(np.median(np.diff(arr_y)))
    step = max(7.0, min(step_x, step_y))

    h, w = wgray.shape[:2]
    x0_line = float(xs[0])
    x18 = float(xs[18])
    y0_line = float(ys[0])
    y18 = float(ys[18])
    margin_bb = 0.22 * step
    max_center_off = 0.40 * step

    out = []
    for s in stones:
        try:
            gx = int(s.get("x"))
            gy = int(s.get("y"))
            color = str(s.get("color", "")).upper()
        except Exception:
            continue
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19 or color not in ("B", "W"):
            continue

        cx = float(xs[gx])
        cy = float(ys[gy])
        rad = int(max(5, round(step * 0.50)))
        yi0 = max(0, int(round(cy)) - rad)
        yi1 = min(h, int(round(cy)) + rad + 1)
        xi0 = max(0, int(round(cx)) - rad)
        xi1 = min(w, int(round(cx)) + rad + 1)
        patch = wgray[yi0:yi1, xi0:xi1].astype(np.float32)
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            out.append(s)
            continue

        pflat = patch.ravel()
        if color == "W":
            base = float(np.percentile(pflat, 58))
            wts = np.maximum(0.0, patch - base)
        else:
            base = float(np.percentile(pflat, 42))
            wts = np.maximum(0.0, base - patch)

        sw = float(np.sum(wts))
        if sw < 1e-3:
            out.append(s)
            continue

        hh, ww = patch.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        mx = float(np.sum(xx * wts) / sw) + xi0
        my = float(np.sum(yy * wts) / sw) + yi0

        off = float(np.hypot(mx - cx, my - cy))
        # 手机图模糊/压缩后质心略偏，略放宽以免误删真子
        if off > max_center_off * 1.12:
            continue

        if mx < x0_line - margin_bb or mx > x18 + margin_bb or my < y0_line - margin_bb or my > y18 + margin_bb:
            continue

        out.append(s)

    return out


def _filter_stones_white_reflection_and_offboard(warped_bgr, wgray, xs, ys, stones):
    """
    进一步去掉：
    1) 交叉点上的强镜面反光（质心仍在格点，但呈尖峰/高拉普拉斯/细长亮条）；
    2) 边界外白子：亮团加权质量多数落在 19 路首末线围成矩形之外（提子、盒盖等）。
    """
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    if not stones or len(xs) != 19 or len(ys) != 19:
        return stones

    step_x = float(np.median(np.diff(np.array(xs, dtype=np.float32))))
    step_y = float(np.median(np.diff(np.array(ys, dtype=np.float32))))
    step = max(7.0, min(step_x, step_y))

    h, w = wgray.shape[:2]
    x_lo = float(xs[0]) - 0.30 * step
    x_hi = float(xs[18]) + 0.30 * step
    y_lo = float(ys[0]) - 0.30 * step
    y_hi = float(ys[18]) + 0.30 * step

    lap = np.abs(cv2.Laplacian(wgray, cv2.CV_32F, ksize=3))

    out = []
    for s in stones:
        try:
            color = str(s.get("color", "")).upper()
            gx = int(s.get("x"))
            gy = int(s.get("y"))
        except Exception:
            continue
        if color != "W":
            out.append(s)
            continue
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            continue

        cx = int(xs[gx])
        cy = int(ys[gy])
        rad = int(max(5, round(step * 0.52)))
        xi0, xi1 = max(0, cx - rad), min(w, cx + rad + 1)
        yi0, yi1 = max(0, cy - rad), min(h, cy + rad + 1)
        patch = wgray[yi0:yi1, xi0:xi1].astype(np.float32)
        if patch.size < 30:
            out.append(s)
            continue

        pflat = patch.ravel()
        thr_w = float(np.percentile(pflat, 80))
        wmask = np.maximum(0.0, patch - thr_w)
        sw = float(np.sum(wmask))
        if sw < 1e-6:
            out.append(s)
            continue

        hh, ww = patch.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        px = xx + xi0
        py = yy + yi0
        inside = (px >= x_lo) & (px <= x_hi) & (py >= y_lo) & (py <= y_hi)
        frac_in = float(np.sum(wmask * inside.astype(np.float32))) / sw
        # 手机压缩后亮团略散，过低阈值会误删真白子
        # 略放宽：压缩/散焦后亮团质心略偏出首末线围成的矩形，真白子曾被误删
        if frac_in < 0.36:
            continue

        pstd = float(np.std(pflat))
        pmax = float(np.max(pflat))
        pmin = float(np.min(pflat))
        peak_ratio = (pmax - pmin) / (pstd + 2.5)

        lc = float(lap[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0.0
        an = _patch_gradient_anisotropy(wgray, cx, cy, int(max(4, round(step * 0.27))))

        thr90 = float(np.percentile(pflat, 90))
        frac_top = float(np.mean(patch >= thr90))

        # 镜面尖峰：拉普拉斯大 + 对比度集中在极少数像素
        if lc > 42.0 and peak_ratio > 18.0:
            continue
        if lc > 68.0:
            continue
        if peak_ratio > 28.0 and an > 1.95:
            continue
        if an > 2.72:
            continue

        # 极小的超高亮核（点状反光）
        if frac_top < 0.036 and lc > 28.0:
            continue

        # 彩色高光边缘（BGR 色度大）且仍呈高光形态
        if warped_bgr is not None and warped_bgr.size > 0:
            bp = warped_bgr[yi0:yi1, xi0:xi1]
            if bp.ndim == 3 and bp.shape[0] >= 4 and bp.shape[1] >= 4:
                b, g, r = cv2.split(bp.astype(np.float32))
                chroma = np.maximum(np.abs(r - g), np.maximum(np.abs(r - b), np.abs(g - b)))
                h0, w0 = chroma.shape
                c0 = chroma[h0 // 4 : 3 * h0 // 4, w0 // 4 : 3 * w0 // 4]
                if c0.size > 0:
                    cm = float(np.mean(c0))
                    if cm > 44.0 and lc > 24.0 and peak_ratio > 14.0:
                        continue

        out.append(s)

    return out


def _apply_stone_post_filters(warped, wgray, xs, ys, fused_stones, *, roboflow_mode=False):
    """空间过滤 + 白子反光过滤；若过滤后为空但融合阶段有子，回退以免手机压缩图被误删光。

    roboflow_mode=True：跳过白子反光启发式与「>70 子按 conf 分位剔除」（Roboflow 白子置信度常整体偏低，
    易被误删；错位应靠网格 2D 吸附与 photometry 合并修正，而非强 conf 截断）。
    另：Roboflow 结果已吸附到交叉点，再做质心空间过滤易把手机/杂乱背景图上的真子整盘删光，故整段跳过。
    """
    if not fused_stones:
        return fused_stones
    if roboflow_mode:
        s_spatial = list(fused_stones)
    else:
        s_spatial = _filter_stones_spatial_board_face(wgray, xs, ys, list(fused_stones))
    if roboflow_mode:
        s_final = list(s_spatial)
    else:
        s_final = _filter_stones_white_reflection_and_offboard(
            warped, wgray, xs, ys, list(s_spatial)
        )
    if len(s_final) == 0 and len(fused_stones) > 0:
        if len(s_spatial) > 0:
            return s_spatial
        return list(fused_stones)
    # 低召回回退：如果反光/离盘过滤把子删得太多，直接回退到更宽松的 s_spatial 或 fused_stones
    # （手机图常见：白子被误判为反光/纹理而被筛掉，导致总数明显偏少）
    if 0 < len(fused_stones) and len(s_final) < max(4, int(round(len(fused_stones) * 0.4))):
        if len(s_spatial) >= len(s_final) and len(s_spatial) > 0:
            return s_spatial
        if len(fused_stones) > 0:
            return list(fused_stones)
    # 在基于 conf 的过滤前先做一次黑白亮度自检，避免反光导致颜色标签整体反转时，
    # “黑白不平衡”策略把正确候选误删掉。
    try:
        # Roboflow 流程已有灰度仲裁；网格略偏时采样点落在木纹上会导致 avgB>avgW 误判并整盘反色
        if s_final and not roboflow_mode:
            black_means = []
            white_means = []
            for st in s_final:
                gx = int(st.get("x"))
                gy = int(st.get("y"))
                color = str(st.get("color", "")).upper()
                if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
                    continue
                cx = int(xs[gx])
                cy = int(ys[gy])
                r = 2
                y0 = max(0, cy - r)
                y1 = min(wgray.shape[0], cy + r + 1)
                x0 = max(0, cx - r)
                x1 = min(wgray.shape[1], cx + r + 1)
                patch = wgray[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                m = float(patch.mean())
                if color == "B":
                    black_means.append(m)
                elif color == "W":
                    white_means.append(m)
            if len(black_means) >= 3 and len(white_means) >= 3:
                avgB = sum(black_means) / float(len(black_means))
                avgW = sum(white_means) / float(len(white_means))
                if avgB > avgW:
                    for st in s_final:
                        c = str(st.get("color", "")).upper()
                        if c == "B":
                            st["color"] = "W"
                        elif c == "W":
                            st["color"] = "B"
    except Exception:
        pass
    # 如果识别出来棋子数量偏多/黑白极不平衡，多半是手机压缩/反光导致的低置信度误检。
    # 仅当结果带 conf 且总数足够大时，按 conf 做一次“抑制低置信度”过滤。
    if (
        not roboflow_mode
        and len(s_final) > 70
        and any(("conf" in st) for st in s_final)
    ):
        try:
            import numpy as np  # type: ignore

            confs = [float(st.get("conf", 0.0) or 0.0) for st in s_final]
            # 用黑白不平衡程度自适应更强过滤
            black_cnt = sum(1 for st in s_final if str(st.get("color", "")).upper() == "B")
            white_cnt = len(s_final) - black_cnt
            total = max(1, black_cnt + white_cnt)
            imbalance = abs(black_cnt - white_cnt) / float(total)
            conf_percentile = 80 if imbalance > 0.35 else (75 if imbalance > 0.25 else 65)
            conf_thr = float(np.percentile(confs, conf_percentile))
            filtered = [st for st in s_final if float(st.get("conf", 0.0) or 0.0) >= conf_thr]
            if filtered and len(filtered) <= len(s_final):
                s_final = filtered
            # 如果仍然过多，再按置信度截断，避免整盘被木纹误判为棋子
            if len(s_final) > 92:
                s_final.sort(key=lambda x: float(x.get("conf", 0.0) or 0.0), reverse=True)
                s_final = s_final[:92]
        except Exception:
            # 兜底：过滤失败则不影响主流程
            pass
    return s_final


def _hough_circle_stone_candidates(wgray, xs, ys):
    """
    参考 gbr/GoScanner 的思路：在拉正棋盘上用圆检测补漏。
    返回 {(gx,gy): {'color':'B'|'W', 'conf':float}}
    """
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    if len(xs) != 19 or len(ys) != 19:
        return {}

    step_x = float(np.median(np.diff(np.array(xs, dtype=np.float32))))
    step_y = float(np.median(np.diff(np.array(ys, dtype=np.float32))))
    step = max(7.0, min(step_x, step_y))

    blur = cv2.GaussianBlur(wgray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(7.0, step * 0.78),
        param1=120,
        param2=15,
        minRadius=max(4, int(round(step * 0.23))),
        maxRadius=max(6, int(round(step * 0.54))),
    )
    if circles is None:
        return {}

    board_mean = float(np.mean(wgray))
    out = {}
    circles = np.round(circles[0, :]).astype(np.int32)
    for c in circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if r <= 1:
            continue
        gx = int(np.argmin(np.abs(np.array(xs, dtype=np.float32) - cx)))
        gy = int(np.argmin(np.abs(np.array(ys, dtype=np.float32) - cy)))
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            continue
        # 圆心需贴近交叉点，否则大概率是假阳性（棋盘纹理/反光）
        if abs(cx - xs[gx]) > step * 0.32 or abs(cy - ys[gy]) > step * 0.32:
            continue

        rr = max(3, int(round(r * 0.78)))
        y0 = max(0, cy - rr)
        y1 = min(wgray.shape[0], cy + rr + 1)
        x0 = max(0, cx - rr)
        x1 = min(wgray.shape[1], cx + rr + 1)
        patch = wgray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        pm = float(np.mean(patch))
        # 简单亮度分色：明显暗 -> 黑；明显亮 -> 白
        if pm <= board_mean - 11:
            color = "B"
            conf = (board_mean - pm) / 30.0
        elif pm >= board_mean + 9:
            an = _patch_gradient_anisotropy(wgray, cx, cy, max(4, int(round(rr * 0.95))))
            if an > 2.85:
                continue
            pstd = float(np.std(patch.astype(np.float32)))
            if pstd > 26.0 and (pm - board_mean) < 24.0:
                continue
            color = "W"
            conf = (pm - board_mean) / 28.0
        else:
            continue
        conf = float(max(0.0, min(1.0, conf)))

        key = (gx, gy)
        prev = out.get(key)
        if prev is None or conf > prev["conf"]:
            out[key] = {"color": color, "conf": conf}
    return out


def _classify_points(wgray, xs, ys, relaxed=False, ultra_relaxed=False):
    import numpy as np  # type: ignore

    h, w = wgray.shape[:2]
    if len(xs) != 19 or len(ys) != 19:
        return []

    def local_step(arr, i):
        if i <= 0:
            return float(arr[1] - arr[0])
        if i >= len(arr) - 1:
            return float(arr[-1] - arr[-2])
        return float((arr[i + 1] - arr[i - 1]) * 0.5)

    # 全局基线只用于兜底限制
    step_x_global = float(np.median(np.diff(np.array(xs, dtype=np.float32))))
    step_y_global = float(np.median(np.diff(np.array(ys, dtype=np.float32))))
    step_global = max(8.0, min(step_x_global, step_y_global))

    yy, xx = np.ogrid[:h, :w]
    feats = []
    for gy in range(19):
        for gx in range(19):
            cx = int(xs[gx])
            cy = int(ys[gy])
            sx_loc = max(6.0, local_step(xs, gx))
            sy_loc = max(6.0, local_step(ys, gy))
            # 关键：按每个交叉点局部网格间距设定采样半径，解决“近大远小”
            step_loc = max(6.0, min(sx_loc, sy_loc))
            step_loc = max(step_global * 0.62, min(step_global * 1.55, step_loc))
            r_center = int(max(3, round(step_loc * 0.21)))
            r_mid = int(max(r_center + 2, round(step_loc * 0.34)))
            r_ring_in = int(max(r_mid + 2, round(step_loc * 0.43)))
            r_ring_out = int(max(r_ring_in + 2, round(step_loc * 0.58)))
            r_blob = int(max(r_mid + 2, round(step_loc * 0.54)))

            border_r = max(r_ring_out, r_blob)
            if cx - border_r < 0 or cy - border_r < 0 or cx + border_r >= w or cy + border_r >= h:
                feats.append((gx, gy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
                continue

            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            center_mask = dist2 <= r_center * r_center
            mid_mask = dist2 <= r_mid * r_mid
            ring_mask = (dist2 >= r_ring_in * r_ring_in) & (dist2 <= r_ring_out * r_ring_out)
            blob_mask = dist2 <= r_blob * r_blob
            center_vals = wgray[center_mask]
            mid_vals = wgray[mid_mask]
            ring_vals = wgray[ring_mask]
            blob_vals = wgray[blob_mask]
            if center_vals.size == 0 or mid_vals.size == 0 or ring_vals.size == 0 or blob_vals.size == 0:
                feats.append((gx, gy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
                continue

            cm = float(center_vals.mean())
            mm = float(mid_vals.mean())
            cs = float(center_vals.std())
            rm = float(ring_vals.mean())
            rs = float(ring_vals.std())
            delta = cm - rm
            # mid_delta: 检查扩大半径后是否仍偏暗/偏亮，抑制星位点误判
            mid_delta = mm - rm
            # 局部自适应暗阈值：随局部纹理/噪声浮动，减少远处小子漏检
            dark_thr = rm - max(7.5, 0.85 * rs + 6.0)
            dark_ratio = float(np.mean(blob_vals <= dark_thr))
            tex = cs + 0.35 * rs
            rs_patch = max(4, int(round(step_loc * 0.24)))
            aniso = _patch_gradient_anisotropy(wgray, cx, cy, rs_patch)
            feats.append((gx, gy, delta, mid_delta, cm, rm, tex, dark_ratio, aniso, cs))

    deltas = np.array([f[2] for f in feats], dtype=np.float32)
    mid_deltas = np.array([f[3] for f in feats], dtype=np.float32)
    textures = np.array([f[6] for f in feats], dtype=np.float32)
    dark_ratios = np.array([f[7] for f in feats], dtype=np.float32)
    abs_delta = np.abs(deltas)

    # 自适应阈值：根据整盘分布决定黑白判定边界
    p15 = float(np.percentile(deltas, 15))
    p85 = float(np.percentile(deltas, 85))
    p55_abs = float(np.percentile(abs_delta, 55))
    tex_med = float(np.median(textures))

    if relaxed:
        # 四路全空时的最后手段：压低分位与绝对门槛，优先召回棋子（略增误检可接受）
        p_lo = float(np.percentile(deltas, 22))
        p_hi = float(np.percentile(deltas, 78))
        p40_abs = float(np.percentile(abs_delta, 42))
        if ultra_relaxed:
            # 超放宽：手机低对比/强压缩时 delta 绝对值偏小
            black_thr = max(-3.0, p_lo - 0.6)
            white_thr = min(3.0, p_hi + 0.6)
            min_abs_thr = max(2.2, p40_abs * 0.70)
        else:
            black_thr = min(-4.0, p_lo - 0.8)
            white_thr = max(4.0, p_hi + 0.8)
            min_abs_thr = max(3.5, p40_abs * 0.85)
    else:
        # 略放宽：手机拍照/斜拍拉正后 delta 往往低于扫描图，过严会大面积漏子
        black_thr = min(-6.2, p15 - 1.5)
        white_thr = max(6.2, p85 + 1.5)
        min_abs_thr = max(4.6, p55_abs * 0.90)

    stones = []
    by_pos = {}
    mid_p25 = float(np.percentile(mid_deltas, 25))
    # 黑子扩大半径后也应明显偏暗；星位点通常只在很小中心偏暗
    black_mid_thr = min(-4.0, mid_p25 - 1.5)
    if relaxed:
        if ultra_relaxed:
            black_mid_thr = max(-2.2, float(np.percentile(mid_deltas, 35)) - 0.6)
        else:
            black_mid_thr = min(-2.5, float(np.percentile(mid_deltas, 35)) - 0.8)
    dark_ratio_thr = max(0.38, float(np.percentile(dark_ratios, 72)) * 0.92)
    if relaxed:
        if ultra_relaxed:
            dark_ratio_thr = max(0.20, float(np.percentile(dark_ratios, 55)) * 0.82)
        else:
            dark_ratio_thr = max(0.26, float(np.percentile(dark_ratios, 58)) * 0.85)
    star_points = {
        (3, 3), (9, 3), (15, 3),
        (3, 9), (9, 9), (15, 9),
        (3, 15), (9, 15), (15, 15),
    }

    for gx, gy, delta, mid_delta, cm, rm, tex, dark_ratio, aniso, cs in feats:
        color = None
        conf = 0.0
        ring_margin = 3 if ultra_relaxed else (4 if relaxed else 5)
        white_ring = 4 if ultra_relaxed else (5 if relaxed else 7)
        # 黑子：中心明显更暗，且纹理不能太平（避免把木纹亮暗慢变化当黑子）
        if delta <= black_thr and abs(delta) >= min_abs_thr and cm < rm - ring_margin:
            # 关键：半径扩大后仍偏暗，才算黑子（过滤星位点）
            if mid_delta > black_mid_thr:
                continue
            # 关键：要求有足够暗色面积，防止把星位小黑点当黑子
            if dark_ratio < dark_ratio_thr:
                continue
            # 星位点再加一道更严格门槛
            if not relaxed and (gx, gy) in star_points:
                if dark_ratio < max(0.46, dark_ratio_thr + 0.10):
                    continue
                if mid_delta > (black_mid_thr - 3.0):
                    continue
            elif relaxed and (gx, gy) in star_points:
                if ultra_relaxed:
                    if dark_ratio < max(0.30, dark_ratio_thr + 0.04):
                        continue
                else:
                    if dark_ratio < max(0.38, dark_ratio_thr + 0.06):
                        continue
            tex_scale = 0.50 if ultra_relaxed else (0.55 if relaxed else 0.60)
            if tex >= tex_med * tex_scale:
                color = "B"
                # 将可分性信号压成一个粗置信度：delta/mid_delta 对比越强越高；暗色面积越大越高
                conf = float(max(0.0, (-delta) + (-mid_delta) * 0.20 + dark_ratio * 6.0 - tex * 0.03))
        # 白子：中心明显更亮，纹理通常较平滑
        elif delta >= white_thr and abs(delta) >= min_abs_thr and cm > rm + white_ring:
            tex_hi = 1.75 if ultra_relaxed else (1.55 if relaxed else 1.42)
            if tex <= tex_med * tex_hi:
                # 网格线强反光：细长亮条各向异性大；镜面高光点：中心方差大但对比未压倒性
                if not relaxed:
                    if aniso > 2.92:
                        continue
                    if aniso > 2.25 and delta <= white_thr + 4.0:
                        continue
                    if cs > max(15.5, tex_med * 0.98) and delta < white_thr + 8.0 and aniso > 1.95:
                        continue
                else:
                    if ultra_relaxed:
                        if aniso > 3.65:
                            continue
                        if aniso > 2.80 and delta <= white_thr + 4.0:
                            continue
                    else:
                        if aniso > 3.15:
                            continue
                        if aniso > 2.45 and delta <= white_thr + 3.0:
                            continue
                color = "W"
                # 白子：中心亮度对比强、且各向异性不过高时更可信
                conf = float(max(0.0, delta + (rm - cm) * 0.05 + (cs * 0.02) - max(0.0, aniso - 2.3) * 0.45))

        if color is not None:
            item = {"x": gx, "y": gy, "color": color, "conf": conf}
            stones.append(item)
            by_pos[(gx, gy)] = item

    # 圆检测补漏：仅补“主分类未识别”的点，尽量不覆盖已有结果
    hough_map = _hough_circle_stone_candidates(wgray, xs, ys)
    for (gx, gy), v in hough_map.items():
        if (gx, gy) in by_pos:
            continue
        # 置信度过低不补，减少误检
        hough_min = 0.28 if relaxed else 0.32
        if float(v.get("conf", 0.0)) < hough_min:
            continue
        stones.append(
            {
                "x": gx,
                "y": gy,
                "color": v["color"],
                "conf": float(v.get("conf", 0.0) or 0.0),
            }
        )

    # 最后兜底：在手机弱光/强压缩下，若前面仍全空，按交叉点亮暗对比强度做小规模召回
    if len(stones) == 0 and relaxed and len(feats) == 361:
        d_arr = np.array([f[2] for f in feats], dtype=np.float32)
        md_arr = np.array([f[3] for f in feats], dtype=np.float32)
        dr_arr = np.array([f[7] for f in feats], dtype=np.float32)
        an_arr = np.array([f[8] for f in feats], dtype=np.float32)
        cs_arr = np.array([f[9] for f in feats], dtype=np.float32)
        abs_arr = np.abs(d_arr)

        p_abs = float(np.percentile(abs_arr, 90))
        p_b = float(np.percentile(d_arr, 10))
        p_w = float(np.percentile(d_arr, 90))
        p_dark = float(np.percentile(dr_arr, 60))
        p_mid = float(np.percentile(md_arr, 45))

        abs_thr = max(2.2, p_abs * 0.62) if ultra_relaxed else max(3.2, p_abs * 0.78)
        dark_ratio_thr_local = max(0.12, p_dark * 0.55) if ultra_relaxed else max(0.18, p_dark * 0.70)
        aniso_thr = 3.35 if ultra_relaxed else 2.95
        cs_min = 26.0 if ultra_relaxed else 22.0
        cs_mult = 1.10 if ultra_relaxed else 1.05

        cand = []
        for gx, gy, delta, mid_delta, cm, rm, tex, dark_ratio, aniso, cs in feats:
            if abs(delta) < abs_thr:
                continue
            color = None
            score = 0.0
            if delta <= p_b and mid_delta <= p_mid and dark_ratio >= dark_ratio_thr_local:
                color = "B"
                score = float((-delta) + (-mid_delta) * 0.45 + dark_ratio * 12.0)
            elif delta >= p_w and aniso <= aniso_thr and cs <= max(cs_min, float(np.percentile(cs_arr, 80)) * cs_mult):
                color = "W"
                score = float(delta - max(0.0, aniso - 1.3) * 3.2)
            if color is not None:
                cand.append((score, gx, gy, color))

        cand.sort(reverse=True, key=lambda x: x[0])
        used = set()
        for score, gx, gy, color in cand[:64]:
            if score <= 0:
                continue
            key = (int(gx), int(gy))
            if key in used:
                continue
            used.add(key)
            stones.append({"x": key[0], "y": key[1], "color": color, "conf": float(score)})

    return stones


def _stones_to_pos_map(stones):
    out = {}
    for s in stones:
        try:
            gx = int(s.get("x"))
            gy = int(s.get("y"))
            color = str(s.get("color", "")).upper()
        except Exception:
            continue
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            continue
        if color not in ("B", "W"):
            continue
        conf = float(s.get("conf", 0.0) or 0.0)
        out[(gx, gy)] = {"color": color, "conf": conf}
    return out


def _classify_points_channel_fusion(base_gray, warped_bgr, xs, ys):
    """
    第二层增强：通道分离 + 投票融合。
    参考 gbr 的经验：蓝通道对黑子更敏感，红通道对白子更稳。
    """
    import cv2  # type: ignore

    b_ch, g_ch, r_ch = cv2.split(warped_bgr)
    b_blur = cv2.GaussianBlur(b_ch, (5, 5), 0)
    g_blur = cv2.GaussianBlur(g_ch, (5, 5), 0)
    r_blur = cv2.GaussianBlur(r_ch, (5, 5), 0)

    # 黑子通道图：提升暗子在木纹背景上的对比（偏蓝更稳）
    black_map = cv2.addWeighted(base_gray, 0.72, b_blur, 0.28, 0)
    # 白子通道图：抑制木纹偏黄，提升亮子边缘（偏红更稳）
    white_map = cv2.addWeighted(base_gray, 0.68, r_blur, 0.24, 0)
    # 综合色图：在高反光场景下补充稳定性
    mixed_map = cv2.addWeighted(base_gray, 0.60, g_blur, 0.20, 0)
    mixed_map = cv2.addWeighted(mixed_map, 1.0, b_blur, 0.10, 0)
    mixed_map = cv2.addWeighted(mixed_map, 1.0, r_blur, 0.10, 0)

    base_stones = _classify_points(base_gray, xs, ys)
    black_stones = _classify_points(black_map, xs, ys)
    white_stones = _classify_points(white_map, xs, ys)
    mixed_stones = _classify_points(mixed_map, xs, ys)

    base_map = _stones_to_pos_map(base_stones)
    black_map_pos = _stones_to_pos_map(black_stones)
    white_map_pos = _stones_to_pos_map(white_stones)
    mixed_map_pos = _stones_to_pos_map(mixed_stones)

    # 基线全空时（手机弱光/强压缩下常见），不能再强依赖「2 票 + 基线回填」，否则融合恒为空
    base_empty = len(base_map) == 0

    all_pos = set(base_map.keys()) | set(black_map_pos.keys()) | set(white_map_pos.keys()) | set(mixed_map_pos.keys())
    fused = []
    for pos in all_pos:
        votes_b = 0
        votes_w = 0
        for src in (base_map, black_map_pos, white_map_pos, mixed_map_pos):
            c = src.get(pos)
            if c and c.get("color") == "B":
                votes_b += 1
            elif c and c.get("color") == "W":
                votes_w += 1

        chosen = None
        # 至少 2 票才采用，尽量抑制单路误判
        if votes_b >= 2 and votes_b > votes_w:
            chosen = "B"
        elif votes_w >= 2 and votes_w > votes_b:
            chosen = "W"
        else:
            # 平票/弱票时保守地使用基线结果，避免引入新噪声
            chosen_item = base_map.get(pos)
            chosen = chosen_item.get("color") if chosen_item else None
            if chosen is None and base_empty:
                # 辅路单票即可采纳（四路里谁多信谁），避免整盘空识别
                if votes_b > votes_w:
                    chosen = "B"
                elif votes_w > votes_b:
                    chosen = "W"

        if chosen in ("B", "W"):
            gx, gy = pos
            conf = 0.0
            for src in (base_map, black_map_pos, white_map_pos, mixed_map_pos):
                c = src.get(pos)
                if c and c.get("color") == chosen:
                    conf = max(conf, float(c.get("conf", 0.0) or 0.0))
            fused.append({"x": gx, "y": gy, "color": chosen, "conf": conf})

    # 防止移动端照片在多路投票下“过度保守”导致空结果：
    # 若融合结果明显偏少，回退到基线识别，并补充少量高一致性点。
    base_cnt = len(base_stones)
    fused_cnt = len(fused)
    if base_cnt > 0 and fused_cnt < max(3, int(round(base_cnt * 0.42))):
        merged = {
            (int(s["x"]), int(s["y"])): {
                "color": str(s["color"]).upper(),
                "conf": float(s.get("conf", 0.0) or 0.0),
            }
            for s in base_stones
        }
        for pos in all_pos:
            if pos in merged:
                continue
            votes_b = 0
            votes_w = 0
            for src in (black_map_pos, white_map_pos, mixed_map_pos):
                c = src.get(pos)
                if c and c.get("color") == "B":
                    votes_b += 1
                elif c and c.get("color") == "W":
                    votes_w += 1
            # 仅补充“非基线三路里至少两票一致”的点
            if votes_b >= 2 and votes_b > votes_w:
                merged[pos] = {"color": "B", "conf": 0.0}
            elif votes_w >= 2 and votes_w > votes_b:
                merged[pos] = {"color": "W", "conf": 0.0}
        out = []
        for (gx, gy), v in merged.items():
            color = v.get("color")
            if color in ("B", "W"):
                out.append({"x": gx, "y": gy, "color": color, "conf": float(v.get("conf", 0.0) or 0.0)})
        return out

    # 基线为空且融合仍为空，但辅路合计有子：再按「多路多数」合并一遍
    if base_cnt == 0 and fused_cnt == 0 and all_pos:
        merged = {}
        for pos in all_pos:
            vb = 0
            vw = 0
            for src in (base_map, black_map_pos, white_map_pos, mixed_map_pos):
                c = src.get(pos)
                if c and c.get("color") == "B":
                    vb += 1
                elif c and c.get("color") == "W":
                    vw += 1
            if vb > vw:
                merged[pos] = {"color": "B", "conf": 0.0}
            elif vw > vb:
                merged[pos] = {"color": "W", "conf": 0.0}
        if merged:
            out_fb = []
            for (gx, gy), v in merged.items():
                out_fb.append({"x": gx, "y": gy, "color": v.get("color"), "conf": float(v.get("conf", 0.0) or 0.0)})
            return out_fb

    # 四路特征全未过阈值时 all_pos 为空；用宽松单路再试一次
    if len(fused) == 0:
        fb = _classify_points(base_gray, xs, ys, relaxed=True)
        if fb:
            return fb
        fb2 = _classify_points(base_gray, xs, ys, relaxed=True, ultra_relaxed=True)
        if fb2:
            return fb2

    return fused


def _read_manual_quad_from_env():
    raw = os.environ.get("BOARD_MANUAL_QUAD", "").strip()
    if not raw:
        return None
    try:
        arr = json.loads(raw)
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != 4:
        return None
    out = []
    for p in arr:
        if not isinstance(p, dict):
            return None
        try:
            x = float(p.get("x"))
            y = float(p.get("y"))
        except Exception:
            return None
        out.append([x, y])
    return out


def _downscale_bgr_for_recognition(img_bgr):
    """
    缩小长边以加速四角搜索、网格与 Roboflow 上传；若环境变量里已有手动四角，则同步缩放坐标。
    BOARD_RECOGNIZER_MAX_LONG_EDGE 默认 1920（可设 0 关闭缩放）。
    """
    import cv2  # type: ignore

    try:
        max_edge = int(os.environ.get("BOARD_RECOGNIZER_MAX_LONG_EDGE", "1920").strip())
    except Exception:
        max_edge = 1920
    if max_edge <= 0:
        return img_bgr
    max_edge = max(960, min(3200, max_edge))

    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return img_bgr

    nw = max(1, int(round(w * max_edge / float(m))))
    nh = max(1, int(round(h * max_edge / float(m))))
    out = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    raw = os.environ.get("BOARD_MANUAL_QUAD", "").strip()
    if raw:
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and len(arr) == 4:
                sx = nw / float(w)
                sy = nh / float(h)
                scaled = []
                for p in arr:
                    if not isinstance(p, dict):
                        scaled = None
                        break
                    scaled.append({"x": float(p.get("x")) * sx, "y": float(p.get("y")) * sy})
                if scaled:
                    os.environ["BOARD_MANUAL_QUAD"] = json.dumps(scaled)
        except Exception:
            pass
    return out


def _try_ultralytics_go_recognition(img_bgr, board_pack=None):
    """
    可选的深度学习分支（Ultralytics YOLO：.pt / .onnx 等，与 Roboflow 无关）：
    1) 目标检测/分割识别棋盘与黑白棋子（若模型含 board 类）；
    2) 通过棋盘四点做单应变换；若无 board 类或检测失败，可用本脚本已算好的四角
       （`board_pack=(best_result, best_quad)`，与 Roboflow 同源的传统四角搜索）；
    3) 将棋子中心映射到 19x19 交叉点并离散化。

    开源权重/项目示例（需自行训练或导出为 YOLO 权重后设 BOARD_DL_MODEL）：
    - https://github.com/GoGame-Recognition-Project/GoGame-Detection （Ultralytics + 透视）
    - https://github.com/zhuoyiyao97/YOLO-GO
    - https://github.com/skolchin/gbr （经典 CV，非 YOLO，可作对照）
    - https://github.com/BigBoxxx/goboard （U-Net 线分割，需自行接管线）

    环境变量摘要：
    - BOARD_DL_MODEL：权重路径（.pt/.onnx，须 ultralytics 可读）
    - BOARD_DL_CLASS_MAP：如 {\"board\":0,\"black\":1,\"white\":2}；仅黑白两类的模型设
      BOARD_DL_HAS_BOARD_CLASS=0，并 {\"black\":0,\"white\":1} 等
    - BOARD_DL_USE_TRADITIONAL_QUAD=1（默认）：模型未检出棋盘时用 `_resolve_best_quad` 的四角
    - BOARD_LOCAL_DL_BEFORE_ROBOFLOW=1：在 detect_board_and_stones 中先于 Roboflow 尝试本分支
    """
    import json as _json

    model_path = os.environ.get("BOARD_DL_MODEL", "").strip()
    if not model_path:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO  # type: ignore
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception:
        return None

    class_map_raw = os.environ.get("BOARD_DL_CLASS_MAP", "").strip()
    if class_map_raw:
        try:
            class_map = _json.loads(class_map_raw)
        except Exception:
            class_map = {}
    else:
        class_map = {}
    has_board_cls = os.environ.get("BOARD_DL_HAS_BOARD_CLASS", "1").strip() != "0"
    board_cls = int(class_map.get("board", 0))
    black_cls = int(class_map.get("black", 1))
    white_cls = int(class_map.get("white", 2))
    conf_thr = float(os.environ.get("BOARD_DL_CONF", "0.25").strip() or 0.25)
    iou_thr = float(os.environ.get("BOARD_DL_IOU", "0.45").strip() or 0.45)

    try:
        model = YOLO(model_path)
        preds = model.predict(source=img_bgr, conf=conf_thr, iou=iou_thr, verbose=False)
        if not preds:
            return None
        pred = preds[0]
    except Exception:
        return None

    board_quad = None
    stone_candidates = []
    dl_quad_source = "model"

    # 1) 优先用分割 mask 恢复棋盘四点（对透视更稳）
    if has_board_cls:
        try:
            if hasattr(pred, "masks") and pred.masks is not None and hasattr(pred, "boxes") and pred.boxes is not None:
                import numpy as np  # type: ignore
                import cv2  # type: ignore

                boxes_cls = pred.boxes.cls.cpu().numpy().astype(int).tolist()
                boxes_conf = pred.boxes.conf.cpu().numpy().astype(float).tolist()
                polys = pred.masks.xy
                best_idx = -1
                best_conf = -1.0
                for i, c in enumerate(boxes_cls):
                    if c == board_cls and i < len(polys) and boxes_conf[i] > best_conf:
                        best_idx = i
                        best_conf = boxes_conf[i]
                if best_idx >= 0:
                    poly = np.array(polys[best_idx], dtype=np.float32)
                    if poly.shape[0] >= 4:
                        rect = cv2.minAreaRect(poly.reshape(-1, 1, 2))
                        board_quad = cv2.boxPoints(rect).astype(np.float32)
        except Exception:
            board_quad = None

        # 2) 回退：用 board bbox（轴对齐），仍可通过后续网格拟合修正
        if board_quad is None:
            try:
                if hasattr(pred, "boxes") and pred.boxes is not None:
                    import numpy as np  # type: ignore

                    boxes_xyxy = pred.boxes.xyxy.cpu().numpy().astype(float)
                    boxes_cls = pred.boxes.cls.cpu().numpy().astype(int)
                    boxes_conf = pred.boxes.conf.cpu().numpy().astype(float)
                    idx = -1
                    best_conf = -1.0
                    for i in range(len(boxes_xyxy)):
                        if int(boxes_cls[i]) == board_cls and float(boxes_conf[i]) > best_conf:
                            idx = i
                            best_conf = float(boxes_conf[i])
                    if idx >= 0:
                        x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
                        board_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            except Exception:
                board_quad = None

    use_trad_quad = os.environ.get("BOARD_DL_USE_TRADITIONAL_QUAD", "1").strip() != "0"
    if board_quad is None and use_trad_quad and board_pack is not None:
        try:
            import numpy as np  # type: ignore

            _bq = board_pack[1]
            if _bq is not None:
                board_quad = _order_quad_points(np.asarray(_bq, dtype=np.float32)).copy()
                dl_quad_source = "traditional_best"
        except Exception:
            pass

    if board_quad is None:
        return None

    # 3) 收集棋子检测框中心
    try:
        if hasattr(pred, "boxes") and pred.boxes is not None:
            boxes_xyxy = pred.boxes.xyxy.cpu().numpy().astype(float)
            boxes_cls = pred.boxes.cls.cpu().numpy().astype(int)
            boxes_conf = pred.boxes.conf.cpu().numpy().astype(float)
            for i in range(len(boxes_xyxy)):
                cls_id = int(boxes_cls[i])
                if cls_id not in (black_cls, white_cls):
                    continue
                x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                cx = float((x1 + x2) * 0.5)
                cy = float((y1 + y2) * 0.5)
                color = "B" if cls_id == black_cls else "W"
                stone_candidates.append({"cx": cx, "cy": cy, "color": color, "conf": float(boxes_conf[i])})
    except Exception:
        stone_candidates = []

    # 4) 单应映射到标准棋盘后，吸附到 19 路交叉点
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        dst_size = _board_warp_dst_size()
        dst = np.array(
            [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
            dtype=np.float32,
        )
        oq = _order_quad_points(np.array(board_quad, dtype=np.float32))
        m = cv2.getPerspectiveTransform(oq, dst)
        warped = cv2.warpPerspective(img_bgr, m, (dst_size, dst_size))
        wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        wgray_eq = _clahe_gray(wgray)
        wblur = cv2.GaussianBlur(wgray, (5, 5), 0)
        wblur_eq = cv2.GaussianBlur(wgray_eq, (5, 5), 0)
        gx0 = cv2.Sobel(wblur, cv2.CV_32F, 1, 0, ksize=3)
        gy0 = cv2.Sobel(wblur, cv2.CV_32F, 0, 1, ksize=3)
        gx1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 0, 1, ksize=3)
        gx = 0.58 * gx0 + 0.42 * gx1
        gy = 0.58 * gy0 + 0.42 * gy1
        sx = np.mean(np.abs(gx), axis=0)
        sy = np.mean(np.abs(gy), axis=1)
        sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 21), 0).reshape(-1)
        sy = cv2.GaussianBlur(sy.reshape(-1, 1), (21, 1), 0).reshape(-1)
        xs = _pick_best_grid_positions(sx)
        ys = _pick_best_grid_positions(sy)
        if xs is None or ys is None:
            margin = int(round(dst_size * 0.08))
            xs = [int(round(margin + i * (dst_size - 2 * margin) / 18.0)) for i in range(19)]
            ys = [int(round(margin + i * (dst_size - 2 * margin) / 18.0)) for i in range(19)]
        else:
            xs = _refine_grid_global_shift_1d(sx, xs)
            ys = _refine_grid_global_shift_1d(sy, ys)

        stones_map = {}
        for s in stone_candidates:
            pt = np.array([[[float(s["cx"]), float(s["cy"])]]], dtype=np.float32)
            wp = cv2.perspectiveTransform(pt, m)[0, 0]
            wx, wy = float(wp[0]), float(wp[1])
            gx_idx = int(np.argmin(np.abs(np.array(xs, dtype=np.float32) - wx)))
            gy_idx = int(np.argmin(np.abs(np.array(ys, dtype=np.float32) - wy)))
            if gx_idx < 0 or gx_idx >= 19 or gy_idx < 0 or gy_idx >= 19:
                continue
            step_x = float(np.median(np.diff(np.array(xs, dtype=np.float32))))
            step_y = float(np.median(np.diff(np.array(ys, dtype=np.float32))))
            step = max(7.0, min(step_x, step_y))
            if abs(wx - xs[gx_idx]) > 0.45 * step or abs(wy - ys[gy_idx]) > 0.45 * step:
                continue
            key = (gx_idx, gy_idx)
            prev = stones_map.get(key)
            if prev is None or float(s["conf"]) > float(prev.get("conf", 0.0)):
                stones_map[key] = {"x": gx_idx, "y": gy_idx, "color": s["color"], "conf": float(s["conf"])}

        stones = list(stones_map.values())
        black_count = sum(1 for s in stones if s["color"] == "B")
        white_count = sum(1 for s in stones if s["color"] == "W")
        return {
            "stones": stones,
            "boardSize": 19,
            "blackCount": int(black_count),
            "whiteCount": int(white_count),
            "debugCandidates": [],
            "debugSummary": {
                "mode": "ultralytics",
                "modelPath": model_path,
                "stoneDetections": int(len(stone_candidates)),
                "snappedStones": int(len(stones)),
                "roboflowKeyPresent": _roboflow_api_key_set(),
                "inferenceSource": "ultralytics_yolo",
                "dlQuadSource": dl_quad_source,
                "boardDlHasBoardClass": bool(has_board_cls),
            },
        }
    except Exception:
        return None


def _roboflow_extract_predictions(payload):
    """从 Roboflow 多种 JSON 形态中取出检测列表。"""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    pl = payload.get("predictions")
    if isinstance(pl, list):
        return pl
    if isinstance(pl, dict):
        for v in pl.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    for key in ("detections", "results", "outputs", "output"):
        v = payload.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
        if isinstance(v, dict):
            pl2 = v.get("predictions")
            if isinstance(pl2, list):
                return pl2
    return []


def _roboflow_alignment_score_at_intersection(wgray, xs, ys, gx, gy, color, step, board_ref):
    """在 (gx,gy) 交叉点处，灰度是否支持该子颜色（相对整盘 ref，越大越好）。"""
    import numpy as np  # type: ignore

    try:
        c = str(color or "").upper()
        if c not in ("B", "W"):
            return 0.0
        h, w = wgray.shape[:2]
        cx = int(xs[gx])
        cy = int(ys[gy])
        r = max(3, int(round(float(step) * 0.34)))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = wgray[y0:y1, x0:x1].astype(np.float32)
        if patch.size < 10:
            return 0.0
        med = float(np.median(patch))
        if c == "B":
            return float(board_ref - med)
        return float(med - board_ref)
    except Exception:
        return 0.0


def _roboflow_best_grid_index_shift(wgray, xs, ys, stones_list, step):
    """
    在 {-1,0,1}^2 上搜索整体索引偏移，使吸附格点与拉正面灰度（黑暗白亮）最一致。
    修正「整盘偏一格」且与 winning 网格同源时仍存在的系统误差。
    """
    import numpy as np  # type: ignore

    if not stones_list or len(xs) != 19 or len(ys) != 19:
        return 0, 0

    mdgx = os.environ.get("BOARD_ROBOFLOW_GRID_SHIFT_DGX", "").strip()
    mdgy = os.environ.get("BOARD_ROBOFLOW_GRID_SHIFT_DGY", "").strip()
    if mdgx != "" or mdgy != "":
        try:
            dgx = int(mdgx) if mdgx != "" else 0
            dgy = int(mdgy) if mdgy != "" else 0
        except Exception:
            return 0, 0
        dgx = int(np.clip(dgx, -2, 2))
        dgy = int(np.clip(dgy, -2, 2))
        return dgx, dgy

    board_ref = float(np.median(wgray.astype(np.float32)))

    best_dgx, best_dgy = 0, 0
    best_rank = None
    shift_tot = {}
    for dgx in (-1, 0, 1):
        for dgy in (-1, 0, 1):
            total = 0.0
            ok = True
            tmp = {}
            for s in stones_list:
                gx = int(s.get("x", -99)) + dgx
                gy = int(s.get("y", -99)) + dgy
                if gx < 0 or gx > 18 or gy < 0 or gy > 18:
                    ok = False
                    break
                k = (gx, gy)
                cf = float(s.get("conf", 0.0) or 0.0)
                if k not in tmp or cf > float(tmp[k].get("conf", 0.0) or 0.0):
                    tmp[k] = dict(s)
                    tmp[k]["x"] = gx
                    tmp[k]["y"] = gy
                    tmp[k]["conf"] = cf
            if not ok:
                continue
            for s in tmp.values():
                total += _roboflow_alignment_score_at_intersection(
                    wgray, xs, ys, int(s["x"]), int(s["y"]), s.get("color"), step, board_ref
                )
            shift_tot[(dgx, dgy)] = float(total)
            zero_pri = 1 if (dgx == 0 and dgy == 0) else 0
            l1 = abs(dgx) + abs(dgy)
            rank = (total, zero_pri, -l1)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_dgx, best_dgy = dgx, dgy
    if best_rank is None:
        return 0, 0
    try:
        idx_margin = float(
            os.environ.get("BOARD_ROBOFLOW_GRID_INDEX_SHIFT_MARGIN", "6.0").strip() or 6.0
        )
    except Exception:
        idx_margin = 6.0
    idx_margin = max(0.0, min(80.0, idx_margin))
    if idx_margin > 0 and (best_dgx, best_dgy) != (0, 0):
        t0 = shift_tot.get((0, 0))
        tb = shift_tot.get((best_dgx, best_dgy))
        if t0 is not None and tb is not None and (float(tb) - float(t0)) < idx_margin:
            best_dgx, best_dgy = 0, 0
    return best_dgx, best_dgy


def _roboflow_apply_index_shift_to_stones_map(stones_map, dgx, dgy):
    """对 Roboflow 吸附结果做整体 (dgx,dgy) 平移，冲突格保留 conf 较大者。"""
    if dgx == 0 and dgy == 0:
        return stones_map
    out = {}
    for s in stones_map.values():
        ngx = int(s.get("x", -1)) + dgx
        ngy = int(s.get("y", -1)) + dgy
        if ngx < 0 or ngx > 18 or ngy < 0 or ngy > 18:
            continue
        k = (ngx, ngy)
        cf = float(s.get("conf", 0.0) or 0.0)
        prev = out.get(k)
        if prev is None or cf > float(prev.get("conf", 0.0) or 0.0):
            out[k] = {"x": ngx, "y": ngy, "color": s.get("color"), "conf": cf}
    return out


def _try_roboflow_go_positions_recognition(image_bytes: bytes, img_bgr, board_pack=None):
    """
    通过 Roboflow Universe 的 Hosted Inference API 识别棋子（black/white），并映射到 19x19 网格。
    棋盘四点 + 网格：应传入 `board_pack=(best_result, best_quad)`（与 `detect_board_and_stones` 中
    单次 `_resolve_best_quad` 结果一致），避免重复跑昂贵的四角评估。

    若 HTTP 返回 403（密钥/权限）：立即返回 None，不再调用传统识别回退包（由上层再走 Ultralytics / 纯传统）。

    启用条件：
      - `ROBOFLOW_API_KEY` 存在（由 server.js 或环境变量传入）
      - `BOARD_ROBOFLOW_MODEL_ENDPOINT` 未设置时同 `DEFAULT_BOARD_ROBOFLOW_GO_POSITIONS_MODEL`（解析为 `go-positions/6`）
      - `BOARD_ROBOFLOW_INFERENCE_HOST` 未设置时依次尝试 `https://detect.roboflow.com`、`https://serverless.roboflow.com`（与 Universe 模型页 Hosted API 说明一致；单 URL 或多 URL 逗号分隔）
      - 可选 `BOARD_ROBOFLOW_DETECT_PATH=go-positions/6` 强制指定相对路径
      - `BOARD_ROBOFLOW_CONFIDENCE` 默认 `0.4`（与 Roboflow 托管推理常见默认一致；可调低以增召回）
      - `BOARD_ROBOFLOW_OVERLAP` 默认 `30`（与 Roboflow SDK 一致，0–100）
      - 可选 `BOARD_ROBOFLOW_CLASS_MAP`：JSON，如 `{"0":"B","1":"W"}` 映射 class_id / 类名
      - `BOARD_ROBOFLOW_ID_BLACK` / `BOARD_ROBOFLOW_ID_WHITE` 默认 `0` / `1`（类名为空时按 id 判黑白）
      - `BOARD_ROBOFLOW_SWAP_BW_IDS=1` 交换上述 id 含义
      - Roboflow 路径：网格吸附用 19×19 欧氏最近交叉点；若 `BOARD_ROBOFLOW_MERGE_TRADITIONAL=1` 则与传统结果合并且同格异色用灰度仲裁；后处理仅做空间过滤（不做白子反光剔除 / 大批量 conf 分位剪枝，避免白子被误删）
      - `BOARD_ROBOFLOW_USE_WARPED_GRID` 默认 `0`：吸附与预览一致，使用 winning 候选的 `gridXs/gridYs`；若设 `1` 会在拉正图上重算线位，可能与四角评分网格差约一格导致整盘偏移
      - `BOARD_ROBOFLOW_AUTO_GRID_SHIFT` 默认 `1`：在 {-1,0,1}^2 上按拉正面灰度选整体 (dgx,dgy)，减轻「整盘仍偏一格」
      - `BOARD_ROBOFLOW_GRID_INDEX_SHIFT_MARGIN` 默认 `6.0`：非零索引平移相对 (0,0) 的灰度总分需至少多这么多才采纳，避免噪声下整盘偏一格
      - 可设 `BOARD_ROBOFLOW_GRID_SHIFT_DGX` / `BOARD_ROBOFLOW_GRID_SHIFT_DGY`（整数，如 0 与 1）强制平移，跳过自动搜索
      - `BOARD_ROBOFLOW_INFER_ON_WARPED` 默认 `0`：与 Universe 网页「上传整图」一致，在原图上调用 API，框中心经单应变换到拉正面再吸附网格；若需仅在 1024² 拉正图上推理可设 `1`
      - `BOARD_ROBOFLOW_MAX_INFER_SIDE` 默认 `2048`：原图推理时长边上限（像素），`0` 表示不缩放（全分辨率上传更慢）
      - `BOARD_ROBOFLOW_GRID_PIXEL_NUDGE` 默认 `1`：亚像素级平移网格以对齐检测点与交叉点（`SPAN`/`STEPS` 可调）
      - `BOARD_ROBOFLOW_GRID_COARSE_MARGIN` 默认 `0.06`：粗整格平移 (±1 步) 相对 (0,0) 的得分需至少高这么多才采纳，否则保持 (0,0)
      - `BOARD_ROBOFLOW_MERGE_TRADITIONAL` 默认 `1`：与传统合并补格；仅展示模型框可设 `0`
      - `BOARD_ROBOFLOW_MERGE_SNAP_RATIO_MAX` 默认 `0.48`：当 吸附数/有效框数 ≥ 此值且吸附数≥`MERGE_SUPPRESS_MIN_SNAPPED`(10) 时**跳过**传统合并，避免子数虚高
      - `BOARD_ROBOFLOW_MERGE_FORCE=1`：强制合并（忽略上述抑制）
      - `BOARD_ROBOFLOW_GRAY_ARBITRATE` 默认 `1`：对 RF 结果用灰度在 B/W 间仲裁
      - `BOARD_ROBOFLOW_PRUNE_OVERDENSE` / `BOARD_ROBOFLOW_PRUNE_IF_OVER`：合并后子数过多时剪掉光度不像子的点
      - `BOARD_ROBOFLOW_QUAD_REPICK` 默认 `1`：当 RF 预测框多但吸附极少时，在 `_quadRankedPacks` 中重选四角，使更多框对齐网格（避免「网格分高但透视错」）
      - `BOARD_ROBOFLOW_QUAD_REPICK_MARGIN` 默认 `2`：新四角吸附数需至少比原四角多这么多才切换
      - `BOARD_ROBOFLOW_QUAD_REPICK_MAX` / `BOARD_ROBOFLOW_QUAD_REPICK_POOL`：参与重选与入库的候选数量上限
      - `BOARD_ROBOFLOW_QUAD_REPICK_MIN_PM_RATIO` 默认 `0.88`：基线吸附很少时，重选四角须在拉正面灰度总分上不低于原四角该比例，否则放弃重选（防「吸附数虚高但透视错」）
      - `BOARD_ROBOFLOW_MIN_PRED_CONFIDENCE` 默认 `0`：仅保留置信度 ≥ 该值的检测框（0–1）；设为 `0.2` 等可进一步抑制弱假阳性
    """
    _roboflow_set_last_error("")

    api_key = _sanitize_env_wrapped_string(os.environ.get("ROBOFLOW_API_KEY", ""))
    if not api_key:
        _roboflow_set_last_error("no_api_key_after_fallback")
        return None

    raw_ep = os.environ.get("BOARD_ROBOFLOW_MODEL_ENDPOINT", "").strip()
    model_endpoint = _sanitize_env_wrapped_string(
        raw_ep or DEFAULT_BOARD_ROBOFLOW_GO_POSITIONS_MODEL
    ).strip("/")
    if not model_endpoint:
        model_endpoint = DEFAULT_BOARD_ROBOFLOW_GO_POSITIONS_MODEL

    black_label_hint = os.environ.get("BOARD_ROBOFLOW_BLACK_LABEL", "").strip().lower() or os.environ.get(
        "BOARD_DL_BLACK_LABEL", "black"
    ).strip().lower()
    white_label_hint = os.environ.get("BOARD_ROBOFLOW_WHITE_LABEL", "").strip().lower() or os.environ.get(
        "BOARD_DL_WHITE_LABEL", "white"
    ).strip().lower()

    class_map = {}
    raw_cm = os.environ.get("BOARD_ROBOFLOW_CLASS_MAP", "").strip()
    if raw_cm:
        try:
            class_map = json.loads(raw_cm)
        except Exception:
            class_map = {}
    if not isinstance(class_map, dict):
        class_map = {}

    try:
        import base64
        import requests  # type: ignore
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception as ie:
        _roboflow_set_last_error(f"import_deps:{ie!r}")
        return None

    # 1) 棋盘四点 + 19x19 网格点（优先使用调用方已算好的结果，避免二次 _resolve_best_quad）
    try:
        if board_pack is not None and isinstance(board_pack, (list, tuple)) and len(board_pack) == 2:
            best_result, best_quad = board_pack
        else:
            best_result, best_quad = _resolve_best_quad(img_bgr)
    except Exception as e:
        _roboflow_set_last_error(f"board_pack_or_quad:{e!r}")
        return None

    xs = best_result.get("gridXs")
    ys = best_result.get("gridYs")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != 19 or len(ys) != 19:
        _roboflow_set_last_error("invalid_grid_xs_ys")
        return None

    dst_size = _board_warp_dst_size()
    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
        dtype="float32",
    )
    oq = _order_quad_points(best_quad.astype(np.float32))
    m = cv2.getPerspectiveTransform(oq, dst)
    warped = cv2.warpPerspective(img_bgr, m, (dst_size, dst_size))
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 默认用 winning 四角候选上的网格（与预览/传统识别同源），避免在拉正图上再跑一套线检测
    # 与 Roboflow 吸附坐标系不一致时，会表现为「整盘偏一格」。
    if os.environ.get("BOARD_ROBOFLOW_USE_WARPED_GRID", "0").strip() == "1":
        ngx, ngy = _grid_xy_from_warped_board(warped)
        if ngx is not None and ngy is not None:
            xs, ys = ngx, ngy

    xs_base = [float(v) for v in xs]
    ys_base = [float(v) for v in ys]
    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    step_x = float(np.median(np.diff(xs_arr)))
    step_y = float(np.median(np.diff(ys_arr)))
    step = max(7.0, min(step_x, step_y))
    snap_mul = float(os.environ.get("BOARD_ROBOFLOW_GRID_SNAP", "0.86").strip() or 0.86)
    snap_mul = max(0.25, min(0.96, snap_mul))
    snap_relaxed = min(0.98, snap_mul + 0.22)
    use_default_id_map = os.environ.get("BOARD_ROBOFLOW_USE_DEFAULT_ID_MAP", "1").strip() != "0"
    try:
        id_black = int(os.environ.get("BOARD_ROBOFLOW_ID_BLACK", "0").strip())
        id_white = int(os.environ.get("BOARD_ROBOFLOW_ID_WHITE", "1").strip())
    except Exception:
        id_black, id_white = 0, 1
    if os.environ.get("BOARD_ROBOFLOW_SWAP_BW_IDS", "").strip() == "1":
        id_black, id_white = id_white, id_black

    infer_on_warped = os.environ.get("BOARD_ROBOFLOW_INFER_ON_WARPED", "0").strip() != "0"

    # 与 Universe 一致：原图推理时 infer_src 为上传用图；若做了长边缩放，检测坐标需乘系数回到 img_bgr 再透视变换
    infer_src = warped if infer_on_warped else img_bgr
    infer_orig_scale_x = 1.0
    infer_orig_scale_y = 1.0
    if not infer_on_warped:
        _mis = os.environ.get("BOARD_ROBOFLOW_MAX_INFER_SIDE", "2048").strip().lower()
        if _mis in ("0", "false", "off", "none"):
            max_side = 0
        else:
            try:
                max_side = int(_mis or "2048")
            except Exception:
                max_side = 2048
        if max_side < 0:
            max_side = 0
        ow, oh = int(img_bgr.shape[1]), int(img_bgr.shape[0])
        infer_src = _roboflow_limit_infer_bgr(infer_src, max_side)
        nw, nh = int(infer_src.shape[1]), int(infer_src.shape[0])
        if nw > 0 and nh > 0:
            infer_orig_scale_x = float(ow) / float(nw)
            infer_orig_scale_y = float(oh) / float(nh)

    # 2) 调用 Roboflow hosted inference（Universe 页与 serverless/detect 文档：POST body 为 base64 字符串）
    try:
        try:
            jpg_q = int(os.environ.get("BOARD_ROBOFLOW_JPEG_QUALITY", "88").strip() or 88)
        except Exception:
            jpg_q = 88
        jpg_q = max(60, min(100, jpg_q))
        enc = cv2.imencode(".jpg", infer_src, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
        if not enc or enc[1] is None:
            _roboflow_set_last_error("imencode_jpg_failed")
            return _roboflow_traditional_fallback_pack(
                best_result,
                warped,
                wgray,
                xs,
                ys,
                model_endpoint,
                mode_tag="roboflow_imencode_failed",
                inference_source="traditional_fallback_roboflow_imencode",
            )

        img_jpg = enc[1].tobytes()

        try:
            conf_q = float(os.environ.get("BOARD_ROBOFLOW_CONFIDENCE", "0.4").strip() or 0.4)
        except Exception:
            conf_q = 0.4
        if conf_q >= 1.0:
            conf_q /= 100.0
        conf_q = max(0.01, min(0.99, conf_q))
        try:
            http_timeout = float(os.environ.get("BOARD_ROBOFLOW_HTTP_TIMEOUT", "90").strip() or 90)
        except Exception:
            http_timeout = 90.0
        http_timeout = max(15.0, min(240.0, http_timeout))
        verify_ssl = os.environ.get("BOARD_ROBOFLOW_VERIFY_SSL", "1").strip() != "0"
        try:
            overlap_q = float(os.environ.get("BOARD_ROBOFLOW_OVERLAP", "30").strip() or 30)
        except Exception:
            overlap_q = 30.0
        overlap_q = max(0.0, min(100.0, overlap_q))

        infer_hosts = _roboflow_inference_hosts()
        b64_str = base64.b64encode(img_jpg).decode("ascii")

        payload = None
        used_endpoint = _sanitize_env_wrapped_string(model_endpoint).strip().strip("/")
        used_infer_host = ""
        last_status = None
        path_opts = _roboflow_detect_inference_path_candidates(model_endpoint)
        if not path_opts:
            _roboflow_set_last_error("cannot_parse_detect_path_from_endpoint")
            return _roboflow_traditional_fallback_pack(
                best_result,
                warped,
                wgray,
                xs,
                ys,
                used_endpoint,
                mode_tag="roboflow_bad_endpoint",
                inference_source="traditional_fallback_roboflow_http",
                extra_summary={"hint": "set BOARD_ROBOFLOW_DETECT_PATH like go-positions/6"},
            )

        for infer_host in infer_hosts:
            infer_host = infer_host.strip().rstrip("/")
            if not infer_host:
                continue
            for rel_path in path_opts:
                rel_clean = rel_path.strip().strip("/")
                base_url = f"{infer_host}/{rel_clean}"
                try:
                    qparams = {
                        "api_key": api_key,
                        "confidence": conf_q,
                        "overlap": overlap_q,
                        "format": "json",
                    }
                    resp = requests.post(
                        base_url,
                        params=qparams,
                        data=b64_str,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=http_timeout,
                        verify=verify_ssl,
                    )
                except Exception as e0:
                    _roboflow_set_last_error(f"infer_post:{e0!r}")
                    last_status = -1
                    continue
                last_status = resp.status_code
                # 403：密钥/权限问题，其它 host/path 通常同样无效；不再用传统识别冒充本分支结果
                if resp.status_code == 403:
                    _roboflow_set_last_error(f"http_403:{(resp.text or '')[:200]}")
                    return None
                if resp.status_code != 200:
                    _roboflow_set_last_error(f"http_{resp.status_code}:{(resp.text or '')[:200]}")
                    continue
                try:
                    pl = resp.json()
                except Exception as je:
                    _roboflow_set_last_error(f"json_decode:{je!r}")
                    continue
                pr = _roboflow_extract_predictions(pl)
                if pr:
                    payload = pl
                    used_endpoint = rel_clean
                    used_infer_host = infer_host
                    break
                if payload is None:
                    payload = pl
                    used_endpoint = rel_clean
                    used_infer_host = infer_host
            if payload is not None and _roboflow_extract_predictions(payload):
                break
        if payload is None:
            if last_status == 403:
                return None
            fb = _roboflow_traditional_fallback_pack(
                best_result,
                warped,
                wgray,
                xs,
                ys,
                used_endpoint,
                mode_tag="roboflow_http_failed",
                inference_source="traditional_fallback_roboflow_http",
                extra_summary={"httpStatus": last_status},
            )
            return fb
        model_endpoint = used_endpoint
    except Exception as ex:
        _roboflow_set_last_error(f"roboflow_http_block:{ex!r}")
        fb = _roboflow_traditional_fallback_pack(
            best_result,
            warped,
            wgray,
            xs,
            ys,
            model_endpoint,
            mode_tag="roboflow_exception",
            inference_source="traditional_fallback_roboflow_exception",
        )
        return fb

    preds = _roboflow_extract_predictions(payload)
    if not preds:
        _roboflow_set_last_error("empty_predictions_after_http_200")
        fb = _roboflow_traditional_fallback_pack(
            best_result,
            warped,
            wgray,
            xs,
            ys,
            model_endpoint,
            mode_tag="roboflow-empty+fallback-traditional",
            inference_source="traditional_fallback_roboflow_empty_preds",
            extra_summary={"predictionCount": 0, "roboflowMapped": 0},
        )
        return fb if fb is not None else None

    img_hw = None
    if isinstance(payload, dict):
        imeta = payload.get("image")
        if isinstance(imeta, list) and imeta:
            imeta = imeta[0]
        if isinstance(imeta, dict) and "width" in imeta and "height" in imeta:
            try:
                img_hw = (float(imeta.get("width")), float(imeta.get("height")))
            except Exception:
                img_hw = None

    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def _extract_center(p):
        # Roboflow 通常用 x,y 表示中心坐标（像素坐标）
        if isinstance(p, dict):
            if "x" in p and "y" in p:
                return _to_float(p.get("x")), _to_float(p.get("y"))
            if "center_x" in p and "center_y" in p:
                return _to_float(p.get("center_x")), _to_float(p.get("center_y"))
            # 也可能是 bbox 四角
            if "left" in p and "top" in p and "width" in p and "height" in p:
                cx = _to_float(p.get("left")) + _to_float(p.get("width")) * 0.5
                cy = _to_float(p.get("top")) + _to_float(p.get("height")) * 0.5
                return cx, cy
            bb = p.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                x1, y1, x2, y2 = map(_to_float, bb[:4])
                return (x1 + x2) * 0.5, (y1 + y2) * 0.5
            if all(k in p for k in ("x_min", "y_min", "x_max", "y_max")):
                return (
                    (_to_float(p.get("x_min")) + _to_float(p.get("x_max"))) * 0.5,
                    (_to_float(p.get("y_min")) + _to_float(p.get("y_max"))) * 0.5,
                )
        return None

    def _roboflow_warp_sample(p):
        """单条检测 → (wx, wy, color, conf) 拉正面坐标；无效则 None。"""
        if not isinstance(p, dict):
            return None
        center = _extract_center(p)
        if center is None:
            return None
        cx, cy = center
        ih, iw = (warped.shape[:2] if infer_on_warped else infer_src.shape[:2])
        if img_hw and img_hw[0] > 1 and img_hw[1] > 1:
            if 0 <= cx <= 1.0 and 0 <= cy <= 1.0:
                cx, cy = cx * img_hw[0], cy * img_hw[1]
            elif max(abs(cx), abs(cy)) > 1.5:
                rw, rh = img_hw[0], img_hw[1]
                if abs(rw - float(iw)) > 0.5 or abs(rh - float(ih)) > 0.5:
                    cx = cx * float(iw) / rw
                    cy = cy * float(ih) / rh
        elif iw > 32 and ih > 32 and 0 <= cx <= 1.0 and 0 <= cy <= 1.0:
            cx, cy = cx * float(iw), cy * float(ih)

        if not infer_on_warped:
            cx = cx * float(infer_orig_scale_x)
            cy = cy * float(infer_orig_scale_y)

        raw_cls = p.get("class")
        if raw_cls is not None and not isinstance(raw_cls, str):
            label_raw = str(raw_cls).strip().lower()
        else:
            label_raw = str(
                raw_cls or p.get("class_name") or p.get("label") or p.get("name") or ""
            ).strip().lower()
        cid = None
        if p.get("class_id") is not None:
            try:
                cid = int(p.get("class_id"))
            except Exception:
                cid = None

        color = None
        if class_map:
            if label_raw and label_raw in class_map:
                v = str(class_map[label_raw]).strip().upper()
                if v in ("B", "W"):
                    color = v
            if color is None and cid is not None:
                ks = str(cid)
                if ks in class_map:
                    v = str(class_map[ks]).strip().upper()
                    if v in ("B", "W"):
                        color = v
        if color is None and label_raw:
            if black_label_hint in label_raw or label_raw.startswith("b"):
                color = "B"
            elif white_label_hint in label_raw or label_raw.startswith("w"):
                color = "W"
            elif label_raw == "b":
                color = "B"
            elif label_raw == "w":
                color = "W"
            elif "黑" in label_raw:
                color = "B"
            elif "白" in label_raw:
                color = "W"
        if color is None and use_default_id_map and cid is not None:
            if cid == id_black:
                color = "B"
            elif cid == id_white:
                color = "W"
        if color is None:
            return None
        conf = _to_float(p.get("confidence") if "confidence" in p else p.get("conf"), 0.0)
        if conf <= 0:
            conf = 0.0
        try:
            if infer_on_warped:
                wx, wy = float(cx), float(cy)
            else:
                pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
                wp = cv2.perspectiveTransform(pt, m)[0, 0]
                wx, wy = float(wp[0]), float(wp[1])
        except Exception:
            return None
        return (wx, wy, str(color).upper(), float(conf))

    rf_samples = []
    for _p in preds:
        _s = _roboflow_warp_sample(_p)
        if _s is not None:
            rf_samples.append(_s)

    try:
        min_pred_conf = float(os.environ.get("BOARD_ROBOFLOW_MIN_PRED_CONFIDENCE", "0").strip() or 0)
    except Exception:
        min_pred_conf = 0.0
    min_pred_conf = max(0.0, min(0.99, min_pred_conf))

    def _samples_with_m(m_mat):
        nonlocal m
        old_m = m
        m = m_mat
        out = []
        for _p in preds:
            s = _roboflow_warp_sample(_p)
            if s is not None:
                out.append(s)
        m = old_m
        return out

    def _tier_fuse_sm(samples_list, gxs, gys):
        xa = np.array(gxs, dtype=np.float32)
        ya = np.array(gys, dtype=np.float32)
        st = max(7.0, min(float(np.median(np.diff(xa))), float(np.median(np.diff(ya)))))

        def _fu(snap_limit):
            sm = {}
            for wx, wy, color, conf in samples_list:
                cf = float(conf)
                if cf > 1.0:
                    cf = min(cf / 100.0, 1.0)
                if cf < min_pred_conf:
                    continue
                try:
                    d2 = (wx - xa[:, np.newaxis]) ** 2 + (wy - ya[np.newaxis, :]) ** 2
                    _ui = np.unravel_index(int(np.argmin(d2)), (19, 19))
                    gx_idx, gy_idx = int(_ui[0]), int(_ui[1])
                    min_d = float(np.sqrt(float(d2[gx_idx, gy_idx])))
                    if min_d > snap_limit * st:
                        continue
                except Exception:
                    continue
                key = (gx_idx, gy_idx)
                prev = sm.get(key)
                if prev is None or cf > float(prev.get("conf", 0.0) or 0.0):
                    sm[key] = {"x": gx_idx, "y": gy_idx, "color": color, "conf": float(cf)}
            return sm

        sm = _fu(snap_mul)
        if not sm:
            sm = _fu(snap_relaxed)
        n_sl = len(samples_list)
        need_more = max(5, int(round(0.22 * max(1, n_sl))))
        if len(sm) < need_more and n_sl >= 3:
            sm2 = _fu(min(1.02, snap_relaxed + 0.14))
            if len(sm2) > len(sm):
                sm = sm2
        if n_sl >= 14 and len(sm) < max(6, int(round(0.15 * n_sl))):
            sm4 = _fu(min(1.14, snap_relaxed + 0.24))
            if len(sm4) > len(sm):
                sm = sm4
        return sm

    def _tier_fuse_len(samples_list, gxs, gys):
        return len(_tier_fuse_sm(samples_list, gxs, gys))

    packs_rb = best_result.get("_quadRankedPacks")
    repick_on = os.environ.get("BOARD_ROBOFLOW_QUAD_REPICK", "1").strip() != "0"
    base_snap_len = _tier_fuse_len(rf_samples, xs, ys)
    n_rf0 = len(rf_samples)
    if (
        repick_on
        and isinstance(packs_rb, list)
        and n_rf0 >= 10
        and base_snap_len < max(6, int(round(0.28 * n_rf0)))
    ):
        try:
            repick_max = int(os.environ.get("BOARD_ROBOFLOW_QUAD_REPICK_MAX", "10").strip() or 10)
        except Exception:
            repick_max = 10
        repick_max = max(4, min(28, repick_max))
        try:
            repick_margin = int(os.environ.get("BOARD_ROBOFLOW_QUAD_REPICK_MARGIN", "2").strip() or 2)
        except Exception:
            repick_margin = 2
        repick_margin = max(1, min(8, repick_margin))
        best_snap_try = base_snap_len
        best_pack = None
        for pk in packs_rb[:repick_max]:
            qn = np.array(pk.get("quad") or [], dtype=np.float32)
            if qn.shape != (4, 2):
                continue
            gx = pk.get("gridXs")
            gy = pk.get("gridYs")
            if not isinstance(gx, list) or not isinstance(gy, list) or len(gx) != 19 or len(gy) != 19:
                continue
            trad_tot = int(pk.get("blackCount", 0)) + int(pk.get("whiteCount", 0))
            # 杂乱背景下错误四角会让传统算法「满盘假子」，不应因 RF 偶然吸附多而选用
            if trad_tot > min(160, n_rf0 + 28) or trad_tot > 200:
                continue
            m_try = cv2.getPerspectiveTransform(_order_quad_points(qn), dst)
            spl = _samples_with_m(m_try)
            if len(spl) < 2:
                continue
            ln = _tier_fuse_len(spl, gx, gy)
            if ln > best_snap_try:
                best_snap_try = ln
                best_pack = pk
        if best_pack is not None and best_snap_try >= base_snap_len + repick_margin:
            if base_snap_len <= 6:
                try:
                    pm_ratio = float(
                        os.environ.get("BOARD_ROBOFLOW_QUAD_REPICK_MIN_PM_RATIO", "0.88").strip()
                        or 0.88
                    )
                except Exception:
                    pm_ratio = 0.88
                pm_ratio = max(0.55, min(1.0, pm_ratio))
                orig_sm_pre = _tier_fuse_sm(rf_samples, xs, ys)
                xsa_o = np.array(xs, dtype=np.float32)
                ysa_o = np.array(ys, dtype=np.float32)
                st_o = max(
                    7.0,
                    min(float(np.median(np.diff(xsa_o))), float(np.median(np.diff(ysa_o)))),
                )
                ref_o = float(np.median(wgray.astype(np.float32)))
                pm_o = 0.0
                for sv in orig_sm_pre.values():
                    pm_o += float(
                        _roboflow_alignment_score_at_intersection(
                            wgray,
                            xs,
                            ys,
                            int(sv["x"]),
                            int(sv["y"]),
                            sv.get("color"),
                            st_o,
                            ref_o,
                        )
                    )
                gx_p = best_pack.get("gridXs") or []
                gy_p = best_pack.get("gridYs") or []
                qn_pm = np.array(best_pack.get("quad") or [], dtype=np.float32)
                m_pm = cv2.getPerspectiveTransform(_order_quad_points(qn_pm), dst)
                spl_pm = _samples_with_m(m_pm)
                warped_pm = cv2.warpPerspective(img_bgr, m_pm, (dst_size, dst_size))
                wgray_pm = cv2.cvtColor(warped_pm, cv2.COLOR_BGR2GRAY)
                sm_c = _tier_fuse_sm(spl_pm, gx_p, gy_p)
                xsa_c = np.array(gx_p, dtype=np.float32)
                ysa_c = np.array(gy_p, dtype=np.float32)
                st_c = max(
                    7.0,
                    min(float(np.median(np.diff(xsa_c))), float(np.median(np.diff(ysa_c)))),
                )
                ref_c = float(np.median(wgray_pm.astype(np.float32)))
                pm_c = 0.0
                for sv in sm_c.values():
                    pm_c += float(
                        _roboflow_alignment_score_at_intersection(
                            wgray_pm,
                            gx_p,
                            gy_p,
                            int(sv["x"]),
                            int(sv["y"]),
                            sv.get("color"),
                            st_c,
                            ref_c,
                        )
                    )
                if len(orig_sm_pre) >= 2 and pm_o > 5.0 and pm_c < pm_o * pm_ratio:
                    best_pack = None
            if best_pack is not None:
                best_quad = np.array(best_pack["quad"], dtype=np.float32)
                oq = _order_quad_points(best_quad)
                m = cv2.getPerspectiveTransform(oq, dst)
                warped = cv2.warpPerspective(img_bgr, m, (dst_size, dst_size))
                wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                xs = [float(v) for v in best_pack["gridXs"]]
                ys = [float(v) for v in best_pack["gridYs"]]
                xs_base = list(xs)
                ys_base = list(ys)
                best_result = {
                    **best_result,
                    "gridXs": xs,
                    "gridYs": ys,
                    "stones": list(best_pack.get("stones") or []),
                    "blackCount": int(best_pack.get("blackCount", 0)),
                    "whiteCount": int(best_pack.get("whiteCount", 0)),
                    "score": float(best_pack.get("score", 0.0)),
                }
                ds_sq = dict(best_result.get("_debugSummary") or {})
                ds_sq["roboflowQuadRepicked"] = True
                ds_sq["roboflowQuadRepickSnaps"] = int(best_snap_try)
                ds_sq["roboflowQuadRepickBaseSnaps"] = int(base_snap_len)
                best_result["_debugSummary"] = ds_sq
                rf_samples = []
                for _p in preds:
                    s = _roboflow_warp_sample(_p)
                    if s is not None:
                        rf_samples.append(s)
                xs_arr = np.array(xs, dtype=np.float32)
                ys_arr = np.array(ys, dtype=np.float32)
                step_x = float(np.median(np.diff(xs_arr)))
                step_y = float(np.median(np.diff(ys_arr)))
                step = max(7.0, min(step_x, step_y))

    # 两阶段网格对齐：先在 {-1,0,1}² 上按 step_x/step_y 平移整格（横纵可不同），
    # 再在 ±亚格距内细调。避免单一大范围 linspace 在 x 上吸到边界（如 dpx≈48、dpy=0）导致纵横向失衡。
    pixel_nudge_dx, pixel_nudge_dy = 0.0, 0.0
    roboflow_coarse_igx, roboflow_coarse_igy = 0, 0
    if (
        os.environ.get("BOARD_ROBOFLOW_GRID_PIXEL_NUDGE", "1").strip() != "0"
        and len(rf_samples) >= 12
    ):
        bx = np.array(xs_base, dtype=np.float32)
        by = np.array(ys_base, dtype=np.float32)
        step_x0 = float(np.median(np.diff(bx)))
        step_y0 = float(np.median(np.diff(by)))
        step_x0 = max(7.0, min(95.0, step_x0))
        step_y0 = max(7.0, min(95.0, step_y0))
        st0 = float(step)
        ref_med = float(np.median(wgray.astype(np.float32)))
        sl_score = float(snap_relaxed)
        min_hits = max(6, int(round(0.22 * len(rf_samples))))
        try:
            fine_mul = float(os.environ.get("BOARD_ROBOFLOW_GRID_PIXEL_NUDGE_SPAN", "0.38").strip() or 0.38)
            fine_mul = max(0.12, min(0.55, fine_mul))
            nsteps_f = int(os.environ.get("BOARD_ROBOFLOW_GRID_PIXEL_NUDGE_STEPS", "9").strip() or 9)
            nsteps_f = max(5, min(17, nsteps_f))
        except Exception:
            fine_mul, nsteps_f = 0.38, 9
        try:
            lam_c = float(os.environ.get("BOARD_ROBOFLOW_GRID_ALIGN_COARSE_LAMBDA", "0.11").strip() or 0.11)
            lam_f = float(os.environ.get("BOARD_ROBOFLOW_GRID_ALIGN_FINE_LAMBDA", "0.017").strip() or 0.017)
        except Exception:
            lam_c, lam_f = 0.11, 0.017

        def _score_rf_on_xy_lists(xs_list, ys_list, reg_pen):
            xsa = np.array(xs_list, dtype=np.float32)
            ysa = np.array(ys_list, dtype=np.float32)
            total = 0.0
            nhit = 0
            for wx, wy, col, cf in rf_samples:
                d2 = (wx - xsa[:, np.newaxis]) ** 2 + (wy - ysa[np.newaxis, :]) ** 2
                _ui2 = np.unravel_index(int(np.argmin(d2)), (19, 19))
                gi, gj = int(_ui2[0]), int(_ui2[1])
                md = float(np.sqrt(float(d2[gi, gj])))
                if md > sl_score * st0:
                    continue
                ph = _roboflow_alignment_score_at_intersection(
                    wgray, xsa.tolist(), ysa.tolist(), gi, gj, col, st0, ref_med
                )
                total += float(ph) + 0.015 * float(cf) - 0.001 * (md / max(st0, 1.0)) ** 2
                nhit += 1
            if nhit < min_hits:
                return None
            sc = total / max(1, nhit)
            return float(sc) - float(reg_pen)

        best_rank = None
        best_igx, best_igy = 0, 0
        coarse_sc = {}
        for igx in (-1, 0, 1):
            for igy in (-1, 0, 1):
                xsa = (bx + float(igx) * step_x0).tolist()
                ysa = (by + float(igy) * step_y0).tolist()
                reg_pen = lam_c * float(igx * igx + igy * igy)
                sc = _score_rf_on_xy_lists(xsa, ysa, reg_pen)
                if sc is None:
                    continue
                coarse_sc[(igx, igy)] = float(sc)
                tie_zero = 1 if igx == 0 and igy == 0 else 0
                rank = (sc, tie_zero, -(abs(igx) + abs(igy)))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_igx, best_igy = igx, igy

        try:
            coarse_margin = float(
                os.environ.get("BOARD_ROBOFLOW_GRID_COARSE_MARGIN", "0.06").strip() or 0.06
            )
        except Exception:
            coarse_margin = 0.06
        coarse_margin = max(0.0, min(0.5, coarse_margin))
        if coarse_margin > 0 and (best_igx, best_igy) != (0, 0):
            s00 = coarse_sc.get((0, 0))
            sb = coarse_sc.get((best_igx, best_igy))
            if s00 is not None and sb is not None and (float(sb) - float(s00)) < coarse_margin:
                best_igx, best_igy = 0, 0

        roboflow_coarse_igx, roboflow_coarse_igy = best_igx, best_igy
        bx2 = bx + float(best_igx) * step_x0
        by2 = by + float(best_igy) * step_y0
        span_f = fine_mul * st0
        best_rank2 = None
        best_dpx, best_dpy = 0.0, 0.0
        for dpx in np.linspace(-span_f, span_f, nsteps_f):
            for dpy in np.linspace(-span_f, span_f, nsteps_f):
                xsa = (bx2 + float(dpx)).tolist()
                ysa = (by2 + float(dpy)).tolist()
                reg_f = lam_f * ((float(dpx) / max(st0, 1.0)) ** 2 + (float(dpy) / max(st0, 1.0)) ** 2)
                sc = _score_rf_on_xy_lists(xsa, ysa, reg_f)
                if sc is None:
                    continue
                tie_zero = 1 if abs(float(dpx)) < 1e-9 and abs(float(dpy)) < 1e-9 else 0
                rank2 = (sc, tie_zero, -(abs(float(dpx)) + abs(float(dpy))))
                if best_rank2 is None or rank2 > best_rank2:
                    best_rank2 = rank2
                    best_dpx, best_dpy = float(dpx), float(dpy)

        if best_rank2 is not None:
            pixel_nudge_dx = float(best_igx) * step_x0 + best_dpx
            pixel_nudge_dy = float(best_igy) * step_y0 + best_dpy
        elif best_rank is not None:
            pixel_nudge_dx = float(best_igx) * step_x0
            pixel_nudge_dy = float(best_igy) * step_y0
        else:
            pixel_nudge_dx, pixel_nudge_dy = 0.0, 0.0

        if abs(pixel_nudge_dx) > 1e-6 or abs(pixel_nudge_dy) > 1e-6:
            xs = [xs_base[i] + pixel_nudge_dx for i in range(19)]
            ys = [ys_base[i] + pixel_nudge_dy for i in range(19)]
            xs_arr = np.array(xs, dtype=np.float32)
            ys_arr = np.array(ys, dtype=np.float32)
            step_x = float(np.median(np.diff(xs_arr)))
            step_y = float(np.median(np.diff(ys_arr)))
            step = max(7.0, min(step_x, step_y))

    # 默认与传统识别合并：RF 在杂乱背景/透视下常大量吸附失败，传统可补漏
    merge_traditional = os.environ.get("BOARD_ROBOFLOW_MERGE_TRADITIONAL", "1").strip() != "0"

    def _fuse_with_snap(snap_limit):
        sm = {}
        for p in preds:
            smpl = _roboflow_warp_sample(p)
            if smpl is None:
                continue
            wx, wy, color, conf = smpl
            cf = float(conf)
            if cf > 1.0:
                cf = min(cf / 100.0, 1.0)
            if cf < min_pred_conf:
                continue
            try:
                d2 = (wx - xs_arr[:, np.newaxis]) ** 2 + (wy - ys_arr[np.newaxis, :]) ** 2
                _ui = np.unravel_index(int(np.argmin(d2)), (19, 19))
                gx_idx, gy_idx = int(_ui[0]), int(_ui[1])
                min_d = float(np.sqrt(float(d2[gx_idx, gy_idx])))
                if min_d > snap_limit * step:
                    continue
            except Exception:
                continue

            key = (gx_idx, gy_idx)
            prev = sm.get(key)
            if prev is None or cf > float(prev.get("conf", 0.0) or 0.0):
                sm[key] = {"x": gx_idx, "y": gy_idx, "color": color, "conf": float(cf)}

        return sm

    stones_map = _fuse_with_snap(snap_mul)
    if not stones_map:
        stones_map = _fuse_with_snap(snap_relaxed)
    # 仍过少：再放宽一档（检测框中心偏离交叉点时）
    n_rf = len(rf_samples)
    need_more = max(5, int(round(0.22 * max(1, n_rf))))
    if len(stones_map) < need_more and n_rf >= 3:
        snap_loose = min(1.02, snap_relaxed + 0.14)
        sm2 = _fuse_with_snap(snap_loose)
        if len(sm2) > len(stones_map):
            stones_map = sm2
    # 预测框很多但仍吸附不足：再放宽一档，优先把 RF 框吸上格，少依赖易误判的传统补格
    if n_rf >= 14 and len(stones_map) < max(6, int(round(0.15 * n_rf))):
        sm4 = _fuse_with_snap(min(1.14, snap_relaxed + 0.24))
        if len(sm4) > len(stones_map):
            stones_map = sm4

    roboflow_shift_dgx, roboflow_shift_dgy = 0, 0
    # 子太少时整格平移搜索不可靠，易把子移出盘外
    if (
        os.environ.get("BOARD_ROBOFLOW_AUTO_GRID_SHIFT", "1").strip() != "0"
        and stones_map
        and len(stones_map) >= 8
    ):
        _dgx, _dgy = _roboflow_best_grid_index_shift(
            wgray, xs, ys, list(stones_map.values()), step
        )
        roboflow_shift_dgx, roboflow_shift_dgy = _dgx, _dgy
        stones_map = _roboflow_apply_index_shift_to_stones_map(stones_map, _dgx, _dgy)

    roboflow_snapped = len(stones_map)
    fused = list(stones_map.values())
    trad_list = best_result.get("stones") or []
    merge_force = os.environ.get("BOARD_ROBOFLOW_MERGE_FORCE", "").strip() == "1"
    try:
        merge_snap_ratio_max = float(
            os.environ.get("BOARD_ROBOFLOW_MERGE_SNAP_RATIO_MAX", "0.48").strip() or 0.48
        )
    except Exception:
        merge_snap_ratio_max = 0.48
    merge_snap_ratio_max = max(0.12, min(0.90, merge_snap_ratio_max))
    try:
        merge_suppress_min_snap = int(
            os.environ.get("BOARD_ROBOFLOW_MERGE_SUPPRESS_MIN_SNAPPED", "10").strip() or 10
        )
    except Exception:
        merge_suppress_min_snap = 10
    merge_suppress_min_snap = max(6, min(30, merge_suppress_min_snap))
    snap_ratio = float(roboflow_snapped) / float(max(1, n_rf))
    merge_suppressed_high_snap = False
    merge_traditional_effective = merge_traditional
    if (
        merge_traditional_effective
        and not merge_force
        and n_rf >= 8
        and roboflow_snapped >= merge_suppress_min_snap
        and snap_ratio >= merge_snap_ratio_max
    ):
        # RF 已吸附绝大部分框时，传统补格极易在错误四角上「填满空位」→ 子数暴涨
        merge_traditional_effective = False
        merge_suppressed_high_snap = True

    trad_fill = _sanitize_env_wrapped_string(
        os.environ.get("BOARD_ROBOFLOW_MERGE_TRAD_FILL", "strict")
    ).lower()
    if trad_fill not in ("strict", "relaxed", "none"):
        trad_fill = "strict"
    merge_fill_upgraded = False
    if merge_traditional_effective and trad_list:
        fused = _merge_roboflow_and_traditional_stones(
            fused,
            trad_list,
            wgray=wgray,
            xs=xs,
            ys=ys,
            step=step,
            traditional_fill_mode=trad_fill,
        )
        # strict 下几乎只有 RF、传统大量被拒：略放宽一档补子，避免回到「整盘空」
        if (
            trad_fill == "strict"
            and len(fused) <= roboflow_snapped + 3
            and len(fused) < max(8, int(round(0.22 * len(trad_list))))
            and n_rf >= 12
            and len(trad_list) >= 10
        ):
            fused_rel = _merge_roboflow_and_traditional_stones(
                list(stones_map.values()),
                trad_list,
                wgray=wgray,
                xs=xs,
                ys=ys,
                step=step,
                traditional_fill_mode="relaxed",
            )
            if len(fused_rel) > len(fused):
                fused = fused_rel
                merge_fill_upgraded = True

    fused = _roboflow_arbitrate_colors_by_gray(fused, wgray, xs, ys, step)
    fused = _roboflow_prune_overdense_rf(fused, wgray, xs, ys, step)

    if not fused:
        _roboflow_set_last_error("roboflow_boxes_unmapped_use_traditional")
        fb = _roboflow_traditional_fallback_pack(
            best_result,
            warped,
            wgray,
            xs,
            ys,
            model_endpoint,
            mode_tag="roboflow_map_failed+fallback-traditional",
            inference_source="traditional_fallback_roboflow_unmapped",
            extra_summary={"predictionCount": int(len(preds)), "roboflowMapped": 0},
        )
        return fb

    stones = _apply_stone_post_filters(warped, wgray, xs, ys, fused, roboflow_mode=True)
    black_count = sum(1 for s in stones if s.get("color") == "B")
    white_count = sum(1 for s in stones if s.get("color") == "W")

    base_dbg = dict(best_result.get("_debugSummary") or {})
    base_dbg.update(
        {
            "mode": (
                "roboflow-hosted+merged"
                if merge_traditional_effective
                else "roboflow-hosted"
            ),
            "modelEndpoint": model_endpoint,
            "predictionCount": int(len(preds)),
            "roboflowMapped": int(roboflow_snapped),
            "stonesAfterMerge": int(len(fused)),
            "roboflowMergeTraditional": bool(merge_traditional_effective),
            "roboflowMergeSuppressedHighSnap": bool(merge_suppressed_high_snap),
            "roboflowSnapRatio": round(float(snap_ratio), 4),
            "roboflowMinPredConfidence": round(float(min_pred_conf), 4),
            "inferenceSource": "roboflow_universe",
            "roboflowKeyPresent": True,
            "roboflowGridShift": [int(roboflow_shift_dgx), int(roboflow_shift_dgy)],
            "roboflowInferSpace": "warped" if infer_on_warped else "original",
            "roboflowInferenceHost": used_infer_host or "",
            "roboflowGridCoarse": [int(roboflow_coarse_igx), int(roboflow_coarse_igy)],
            "roboflowGridPixelNudge": [
                round(float(pixel_nudge_dx), 4),
                round(float(pixel_nudge_dy), 4),
            ],
            "roboflowMergeTradFill": trad_fill,
            "roboflowMergeTradFillUpgraded": bool(merge_fill_upgraded),
        }
    )

    return {
        "stones": stones,
        "boardSize": 19,
        "blackCount": int(black_count),
        "whiteCount": int(white_count),
        "debugCandidates": best_result.get("_debugCandidates", []),
        "debugSummary": base_dbg,
    }


def _generate_manual_quad_variants(manual_quad):
    import numpy as np  # type: ignore

    q = _order_quad_points(np.array(manual_quad, dtype=np.float32))
    c = np.mean(q, axis=0)
    w = float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) * 0.5
    h = float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) * 0.5
    dx_base = max(1.5, 0.006 * w)
    dy_base = max(1.5, 0.006 * h)

    variants = []
    # A. 全局小幅微调（点得比较准时）
    scales = [0.97, 0.99, 1.00, 1.01, 1.03]
    shifts = [-0.5, 0.0, 0.5]
    for s in scales:
        qs = c + (q - c) * s
        for sx in shifts:
            for sy in shifts:
                qv = qs.copy()
                qv[:, 0] += sx * dx_base
                qv[:, 1] += sy * dy_base
                variants.append(qv.astype(np.float32))

    # B. 透视/梯形微调：上下边、左右边独立偏移（解决“棋盘是梯形”）
    edge_shifts_x = [-1.0, 0.0, 1.0]
    edge_shifts_y = [-1.0, 0.0, 1.0]
    for top_dy in edge_shifts_y:
        for bottom_dy in edge_shifts_y:
            for left_dx in edge_shifts_x:
                for right_dx in edge_shifts_x:
                    qv = q.copy()
                    # top edge
                    qv[0, 1] += top_dy * dy_base
                    qv[1, 1] += top_dy * dy_base
                    # bottom edge
                    qv[3, 1] += bottom_dy * dy_base
                    qv[2, 1] += bottom_dy * dy_base
                    # left edge
                    qv[0, 0] += left_dx * dx_base
                    qv[3, 0] += left_dx * dx_base
                    # right edge
                    qv[1, 0] += right_dx * dx_base
                    qv[2, 0] += right_dx * dx_base
                    variants.append(qv.astype(np.float32))

    # C. 斜向透视：上/下边各自水平错切，左/右边各自垂直错切
    shear_vals = [-0.8, 0.0, 0.8]
    for top_dx in shear_vals:
        for bottom_dx in shear_vals:
            for left_dy in shear_vals:
                for right_dy in shear_vals:
                    qv = q.copy()
                    qv[0, 0] += top_dx * dx_base
                    qv[1, 0] += top_dx * dx_base
                    qv[3, 0] += bottom_dx * dx_base
                    qv[2, 0] += bottom_dx * dx_base
                    qv[0, 1] += left_dy * dy_base
                    qv[3, 1] += left_dy * dy_base
                    qv[1, 1] += right_dy * dy_base
                    qv[2, 1] += right_dy * dy_base
                    variants.append(qv.astype(np.float32))

    # 去重（避免大量近似重复候选）
    dedup = []
    # 保底把原始手动四角也放进去
    variants.append(q.astype(np.float32))

    for v in variants:
        # 过滤异常翻转/退化四边形
        area = abs(float(
            v[0, 0] * v[1, 1] - v[1, 0] * v[0, 1] +
            v[1, 0] * v[2, 1] - v[2, 0] * v[1, 1] +
            v[2, 0] * v[3, 1] - v[3, 0] * v[2, 1] +
            v[3, 0] * v[0, 1] - v[0, 0] * v[3, 1]
        ) * 0.5)
        if area < 64.0:
            continue
        keep = True
        for d in dedup:
            if float(np.mean(np.abs(v - d))) < 0.45:
                keep = False
                break
        if keep:
            dedup.append(v)
    try:
        cap = int(os.environ.get("BOARD_MANUAL_VARIANT_CAP", "36").strip())
    except Exception:
        cap = 36
    cap = max(8, min(120, cap))
    return dedup[:cap]


def _board_detect_angles_from_env():
    raw = os.environ.get("BOARD_DETECT_ANGLES", "").strip()
    if raw:
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and len(arr) > 0:
                return [float(x) for x in arr]
        except Exception:
            pass
    # 默认 5 个角度（×3 种检测器 ≈15 个候选），显著少于原先 7 角度×3
    return [0.0, -15.0, 15.0, -8.0, 8.0]


def _grid_xy_from_warped_board(warped_bgr):
    """
    在已拉正的棋盘图（通常 1024²）上仅用线梯度估计 19 路网格线位置。
    仅当 `BOARD_ROBOFLOW_USE_WARPED_GRID=1` 时用于 Roboflow 吸附；与 winning 候选上已有网格可能差约一格，
    默认改为使用 `best_result` 的 gridXs/gridYs 以免整盘错位。
    """
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    if warped_bgr is None or warped_bgr.size == 0 or warped_bgr.ndim != 3:
        return None, None
    h, w = warped_bgr.shape[:2]
    if h < 256 or w < 256:
        return None, None

    wgray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    wgray_eq = _clahe_gray(wgray)
    wblur = cv2.GaussianBlur(wgray, (5, 5), 0)
    wblur_eq = cv2.GaussianBlur(wgray_eq, (5, 5), 0)
    gx0 = cv2.Sobel(wblur, cv2.CV_32F, 1, 0, ksize=3)
    gy0 = cv2.Sobel(wblur, cv2.CV_32F, 0, 1, ksize=3)
    gx1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 0, 1, ksize=3)
    gx = 0.58 * gx0 + 0.42 * gx1
    gy = 0.58 * gy0 + 0.42 * gy1
    sx = np.mean(np.abs(gx), axis=0)
    sy = np.mean(np.abs(gy), axis=1)
    sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 21), 0).reshape(-1)
    sy = cv2.GaussianBlur(sy.reshape(-1, 1), (21, 1), 0).reshape(-1)

    xs = _pick_best_grid_positions(sx)
    ys = _pick_best_grid_positions(sy)
    dst_size = int(h)
    if xs is None or ys is None:
        margin = int(round(dst_size * 0.08))
        step_fb = (dst_size - 2 * margin) / 18.0
        xs = [int(round(margin + i * step_fb)) for i in range(19)]
        ys = [int(round(margin + i * step_fb)) for i in range(19)]
    else:
        xs_r = _refine_grid_equal_spacing_1d(sx, xs)
        ys_r = _refine_grid_equal_spacing_1d(sy, ys)
        if xs_r is not None:
            xs = xs_r
        if ys_r is not None:
            ys = ys_r
        xs = _refine_grid_global_shift_1d(sx, xs)
        ys = _refine_grid_global_shift_1d(sy, ys)

    if len(xs) != 19 or len(ys) != 19:
        return None, None
    return xs, ys


def _photometry_traditional_merge_ok(
    wgray, xs, ys, gx, gy, step, claimed_color, *, relaxed: bool = False
):
    """
    Roboflow 与传统合并时：仅当拉正面在该交叉点呈现「明显的棋子状」对比时才采纳传统补格。
    抑制木纹、星位小黑点、阴影等在杂乱背景下被传统算法误判的黑子。
    relaxed=True 时略放宽，仅在 strict 全被拒且 RF 吸附极少时由上层选用。
    """
    import numpy as np  # type: ignore

    c = str(claimed_color or "").upper()
    if c not in ("B", "W") or wgray is None or len(xs) != 19 or len(ys) != 19:
        return False
    try:
        gx = int(gx)
        gy = int(gy)
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            return False
        h, w = wgray.shape[:2]
        cx = int(xs[gx])
        cy = int(ys[gy])
        step_loc = max(8.0, min(95.0, float(step)))
        r_center = int(max(3, round(step_loc * 0.21)))
        r_mid = int(max(r_center + 2, round(step_loc * 0.34)))
        r_ring_in = int(max(r_mid + 2, round(step_loc * 0.43)))
        r_ring_out = int(max(r_ring_in + 2, round(step_loc * 0.58)))
        border_r = r_ring_out
        if cx - border_r < 0 or cy - border_r < 0 or cx + border_r >= w or cy + border_r >= h:
            return False
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        center_mask = dist2 <= r_center * r_center
        mid_mask = dist2 <= r_mid * r_mid
        ring_mask = (dist2 >= r_ring_in * r_ring_in) & (dist2 <= r_ring_out * r_ring_out)
        cm = float(np.mean(wgray[center_mask]))
        mm = float(np.mean(wgray[mid_mask]))
        rm = float(np.mean(wgray[ring_mask]))
        delta = cm - rm
        mid_delta = mm - rm
        star_points = {
            (3, 3), (9, 3), (15, 3),
            (3, 9), (9, 9), (15, 9),
            (3, 15), (9, 15), (15, 15),
        }
        if relaxed:
            if c == "B":
                thr_d = -9.5 if (gx, gy) in star_points else -7.5
                return (
                    delta <= thr_d
                    and abs(delta) >= 6.5
                    and cm < rm - 5.0
                    and mid_delta <= -2.8
                )
            thr_d = 9.5 if (gx, gy) in star_points else 7.5
            return (
                delta >= thr_d
                and abs(delta) >= 6.5
                and cm > rm + 5.0
                and mid_delta >= 2.8
            )
        if c == "B":
            thr_d = -11.0 if (gx, gy) in star_points else -9.0
            return (
                delta <= thr_d
                and abs(delta) >= 8.5
                and cm < rm - 6.5
                and mid_delta <= -4.0
            )
        thr_d = 11.0 if (gx, gy) in star_points else 9.0
        return (
            delta >= thr_d
            and abs(delta) >= 8.5
            and cm > rm + 6.5
            and mid_delta >= 4.0
        )
    except Exception:
        return False


def _photometry_stone_color_hint(wgray, xs, ys, gx, gy, step):
    """
    在拉正灰度图上用局部相对整盘中值推断该交叉点更像黑子还是白子。
    用于 Roboflow 与同格传统识别颜色不一致时的仲裁（不依赖 conf，避免白子整体低分被压制）。
    """
    import numpy as np  # type: ignore

    try:
        if wgray is None or len(xs) != 19 or len(ys) != 19:
            return None
        gx = int(gx)
        gy = int(gy)
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            return None
        h, w = wgray.shape[:2]
        cx = int(xs[gx])
        cy = int(ys[gy])
        r = max(4, int(round(float(step) * 0.38)))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = wgray[y0:y1, x0:x1].astype(np.float32)
        if patch.size < 12:
            return None
        ref = float(np.median(wgray.astype(np.float32)))
        pflat = patch.ravel()
        pmid = float(np.percentile(pflat, 50))
        p10 = float(np.percentile(pflat, 10))
        p90 = float(np.percentile(pflat, 90))
        # 黑子：整体偏暗，且暗部很低
        if p90 < ref - 5.0 and pmid < ref - 3.0:
            return "B"
        if pmid < ref - 16.0:
            return "B"
        # 白子：整体偏亮，且亮部很高
        if p10 > ref + 7.0 and pmid > ref + 4.0:
            return "W"
        if pmid > ref + 16.0:
            return "W"
    except Exception:
        return None
    return None


def _roboflow_arbitrate_colors_by_gray(fused_list, wgray, xs, ys, step):
    """
    RF 框颜色偶错时，用拉正面中心/环带光度在 B/W 间二选一（relaxed 门槛）。
    """
    if not fused_list or os.environ.get("BOARD_ROBOFLOW_GRAY_ARBITRATE", "1").strip() == "0":
        return fused_list
    if wgray is None or len(xs) != 19 or len(ys) != 19:
        return fused_list
    st_step = float(step) if step is not None else 40.0
    out = []
    for s in fused_list:
        if not s or str(s.get("color", "")).upper() not in ("B", "W"):
            continue
        gx = int(s.get("x", -1))
        gy = int(s.get("y", -1))
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            continue
        c0 = str(s.get("color")).upper()
        ob = _photometry_traditional_merge_ok(wgray, xs, ys, gx, gy, st_step, "B", relaxed=True)
        ow = _photometry_traditional_merge_ok(wgray, xs, ys, gx, gy, st_step, "W", relaxed=True)
        c = c0
        if c0 == "B" and ow and not ob:
            c = "W"
        elif c0 == "W" and ob and not ow:
            c = "B"
        item = dict(s)
        item["color"] = c
        out.append(item)
    return out if out else list(fused_list)


def _roboflow_prune_overdense_rf(fused_list, wgray, xs, ys, step):
    """
    合并后子数异常多时，去掉「光度上连自己声称的颜色都不像棋子」的格点（常为木纹/反光误吸附）。
    """
    if not fused_list or wgray is None or len(xs) != 19 or len(ys) != 19:
        return fused_list
    if os.environ.get("BOARD_ROBOFLOW_PRUNE_OVERDENSE", "1").strip() == "0":
        return fused_list
    try:
        over_n = int(os.environ.get("BOARD_ROBOFLOW_PRUNE_IF_OVER", "28").strip() or 28)
    except Exception:
        over_n = 28
    over_n = max(22, min(48, over_n))
    if len(fused_list) < over_n:
        return fused_list
    st_step = float(step) if step is not None else 40.0
    pr = []
    for s in fused_list:
        if not s or str(s.get("color", "")).upper() not in ("B", "W"):
            continue
        gx = int(s.get("x", -1))
        gy = int(s.get("y", -1))
        if gx < 0 or gx >= 19 or gy < 0 or gy >= 19:
            continue
        cc = str(s.get("color")).upper()
        if _photometry_traditional_merge_ok(wgray, xs, ys, gx, gy, st_step, cc, relaxed=True):
            pr.append(s)
    # 避免剪过头：若剩太少则放弃本次剪枝
    if len(pr) < max(10, int(round(0.42 * len(fused_list)))):
        return fused_list
    return pr


def _merge_roboflow_and_traditional_stones(
    robo_stones,
    trad_stones,
    wgray=None,
    xs=None,
    ys=None,
    step=None,
    *,
    traditional_fill_mode: str = "strict",
):
    """同一交叉点 Roboflow 优先；未覆盖格用传统补漏（strict/relaxed/none）；若同色冲突则用灰度仲裁。"""
    merged = {}
    for s in robo_stones or []:
        if not s or s.get("color") not in ("B", "W"):
            continue
        k = (int(s.get("x")), int(s.get("y")))
        merged[k] = {"x": k[0], "y": k[1], "color": s["color"]}

    import numpy as np  # type: ignore

    st_step = step
    if st_step is None and xs is not None and len(xs) == 19:
        try:
            st_step = float(np.median(np.diff(np.array(xs, dtype=np.float32))))
        except Exception:
            st_step = 40.0
    if st_step is None:
        st_step = 40.0

    for s in trad_stones or []:
        if not s or s.get("color") not in ("B", "W"):
            continue
        k = (int(s.get("x")), int(s.get("y")))
        if k not in merged:
            mode = (traditional_fill_mode or "strict").strip().lower()
            if mode not in ("strict", "relaxed", "none"):
                mode = "strict"
            if mode != "none":
                if wgray is None or xs is None or ys is None:
                    continue
                rel = mode == "relaxed"
                if not _photometry_traditional_merge_ok(
                    wgray, xs, ys, k[0], k[1], st_step, s["color"], relaxed=rel
                ):
                    continue
            merged[k] = {"x": k[0], "y": k[1], "color": s["color"]}
            continue
        rc = str(merged[k].get("color", "")).upper()
        tc = str(s.get("color", "")).upper()
        if rc == tc:
            continue
        hint = _photometry_stone_color_hint(wgray, xs, ys, k[0], k[1], st_step)
        if hint in ("B", "W"):
            merged[k] = {"x": k[0], "y": k[1], "color": hint}
    return list(merged.values())


def _evaluate_quad_candidate(img, quad, dst_size=None):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    if dst_size is None:
        dst_size = _board_warp_dst_size()

    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(_order_quad_points(quad.astype("float32")), dst)
    warped = cv2.warpPerspective(img, m, (dst_size, dst_size))
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    wgray_eq = _clahe_gray(wgray)
    wblur = cv2.GaussianBlur(wgray, (5, 5), 0)
    wblur_eq = cv2.GaussianBlur(wgray_eq, (5, 5), 0)
    # 光照融合图：兼顾原始纹理与均衡后的暗亮细节
    wproc = cv2.addWeighted(wblur, 0.60, wblur_eq, 0.40, 0)

    # 网格定位融合两路梯度，提升斜拍/阴影下线条稳定性
    gx0 = cv2.Sobel(wblur, cv2.CV_32F, 1, 0, ksize=3)
    gy0 = cv2.Sobel(wblur, cv2.CV_32F, 0, 1, ksize=3)
    gx1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(wblur_eq, cv2.CV_32F, 0, 1, ksize=3)
    gx = 0.58 * gx0 + 0.42 * gx1
    gy = 0.58 * gy0 + 0.42 * gy1
    sx = np.mean(np.abs(gx), axis=0)
    sy = np.mean(np.abs(gy), axis=1)
    sx = cv2.GaussianBlur(sx.reshape(1, -1), (1, 21), 0).reshape(-1)
    sy = cv2.GaussianBlur(sy.reshape(-1, 1), (21, 1), 0).reshape(-1)

    xs = _pick_best_grid_positions(sx)
    ys = _pick_best_grid_positions(sy)
    used_fallback = False
    if xs is None or ys is None:
        used_fallback = True
        margin = int(round(dst_size * 0.08))
        xs = [int(round(margin + i * (dst_size - 2 * margin) / 18.0)) for i in range(19)]
        ys = [int(round(margin + i * (dst_size - 2 * margin) / 18.0)) for i in range(19)]
    else:
        xs_r = _refine_grid_equal_spacing_1d(sx, xs)
        ys_r = _refine_grid_equal_spacing_1d(sy, ys)
        if xs_r is not None:
            xs = xs_r
        if ys_r is not None:
            ys = ys_r

    xs = _refine_grid_global_shift_1d(sx, xs)
    ys = _refine_grid_global_shift_1d(sy, ys)

    fused = _classify_points_channel_fusion(wproc, warped, xs, ys)
    if len(fused) == 0:
        for gray_try in (wproc, wgray, wblur):
            fused = _classify_points(gray_try, xs, ys, relaxed=True)
            if fused:
                break
    if len(fused) == 0:
        for gray_try in (wproc, wgray, wblur):
            fused = _classify_points(gray_try, xs, ys, relaxed=True, ultra_relaxed=True)
            if fused:
                break
    # 低召回场景：fused 虽然有子但数量明显不足时，再做一轮 ultra_relaxed 召回合并
    # （手机压缩/反光导致 delta 对比度偏低时，容易漏掉一部分子。）
    if 0 < len(fused) < 110:
        merged = {(int(s.get("x")), int(s.get("y"))): s for s in fused if s and s.get("color") in ("B", "W")}
        for gray_try in (wproc, wgray, wblur):
            extra = _classify_points(gray_try, xs, ys, relaxed=True, ultra_relaxed=True)
            if not extra:
                continue
            for es in extra:
                if not es or es.get("color") not in ("B", "W"):
                    continue
                k = (int(es.get("x")), int(es.get("y")))
                conf = float(es.get("conf", 0.0) or 0.0)
                prev = merged.get(k)
                if prev is None or conf > float(prev.get("conf", 0.0) or 0.0):
                    merged[k] = es
        fused = list(merged.values())
    # 补漏：无论 fused 多少，都把 Hough 圆检测结果并入（手机漏检时尤其有效）
    hm = _hough_circle_stone_candidates(wgray, xs, ys)
    if hm:
        merged_map = {(int(s.get("x")), int(s.get("y"))): s for s in fused if s and s.get("color") in ("B", "W")}
        for (gx, gy), v in hm.items():
            key = (int(gx), int(gy))
            conf_h = float(v.get("conf", 0.0) or 0.0)
            if conf_h < 0.13:
                continue
            cand_item = {"x": key[0], "y": key[1], "color": v["color"], "conf": conf_h}
            prev = merged_map.get(key)
            if prev is None:
                merged_map[key] = cand_item
            else:
                prev_conf = float(prev.get("conf", 0.0) or 0.0)
                # 若原结果置信度低，则用 Hough 覆盖
                if conf_h > prev_conf + 0.06:
                    merged_map[key] = cand_item
        fused = list(merged_map.values())
    if len(fused) == 0:
        hm = _hough_circle_stone_candidates(wgray, xs, ys)
        fused = [
            {
                "x": int(gx),
                "y": int(gy),
                "color": v["color"],
                "conf": float(v.get("conf", 0.0) or 0.0),
            }
            for (gx, gy), v in hm.items()
            if float(v.get("conf", 0.0) or 0.0) >= 0.13
        ]

    stones = _apply_stone_post_filters(warped, wgray, xs, ys, fused)
    black_count = sum(1 for s in stones if s["color"] == "B")
    white_count = sum(1 for s in stones if s["color"] == "W")
    total = black_count + white_count

    # 网格质量分：线强度 + 间距规律（用于解决手动点击不准导致偏一格）
    dx = np.diff(np.array(xs, dtype=np.float32))
    dy = np.diff(np.array(ys, dtype=np.float32))
    reg_x = float(np.std(dx) / (np.mean(dx) + 1e-6))
    reg_y = float(np.std(dy) / (np.mean(dy) + 1e-6))
    line_strength = float(np.mean(sx[np.array(xs, dtype=np.int32)]) + np.mean(sy[np.array(ys, dtype=np.int32)]))
    grid_quality = line_strength / (0.10 + reg_x + reg_y)
    if used_fallback:
        grid_quality *= 0.55

    plausibility = 1.0
    if total > 280:
        plausibility *= 0.12
    elif total > 220:
        plausibility *= 0.45
    elif total == 0:
        # 明确抑制“网格像棋盘但子力全空”的候选，避免在多候选中误胜出
        plausibility *= 0.18
    elif total < 2:
        plausibility *= 0.65
    # 低子数时的“黑白不平衡”惩罚适当放松：手机漏检更常表现为一方漏得更多。
    diff_bw = abs(black_count - white_count)
    denom = float(total + (16 if total < 25 else 8))
    stone_balance = 1.0 - min(0.85, diff_bw / max(1.0, denom))
    score = grid_quality * (0.65 + 0.35 * plausibility) * (0.75 + 0.25 * stone_balance)
    # 低召回场景轻微鼓励“检测到更多子”的候选
    if 0 < total < 60:
        total_factor = 0.7 + 0.3 * min(1.0, float(total) / 30.0)
        score *= float(total_factor)

    return {
        "stones": stones,
        "blackCount": black_count,
        "whiteCount": white_count,
        "score": float(score),
        "gridXs": xs,
        "gridYs": ys,
    }


def _resolve_best_quad(img):
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    manual_quad_raw = _read_manual_quad_from_env()
    quad_candidates = []
    manual_mode = False
    manual_reference_quad = None
    if manual_quad_raw is not None:
        manual_quad = np.array(manual_quad_raw, dtype=np.float32)
        if manual_quad.shape == (4, 2):
            manual_mode = True
            manual_reference_quad = _order_quad_points(manual_quad.copy())
            quad_candidates.extend(_generate_manual_quad_variants(manual_reference_quad))
    else:
        # 斜拍场景：棋盘整体可能是“斜着”的。对多角度旋转图分别检测，再映射回原图。
        angles = _board_detect_angles_from_env()
        for ang in angles:
            rot_gray, _, inv_aff = _rotate_image_keep_bounds(gray, ang)
            primary_quad = _detect_board_quad(rot_gray)
            hough_quad = _detect_board_quad_hough(rot_gray)
            scan_quad = _detect_board_quad_window_scan(rot_gray)

            if primary_quad is not None:
                quad_candidates.append(_map_quad_affine(primary_quad, inv_aff))
            if hough_quad is not None:
                quad_candidates.append(_map_quad_affine(hough_quad, inv_aff))
            if scan_quad is not None:
                quad_candidates.append(_map_quad_affine(scan_quad, inv_aff))

    if not quad_candidates:
        fail("未检测到棋盘边框，请上传完整且清晰的棋盘照片")

    try:
        max_eval = int(os.environ.get("BOARD_MAX_QUAD_EVALUATIONS", "18").strip())
    except Exception:
        max_eval = 18
    max_eval = max(4, min(120, max_eval))
    if len(quad_candidates) > max_eval:
        quad_candidates = quad_candidates[:max_eval]

    best_result = None
    best_score = -1.0
    best_total = -1
    best_quad = None
    best_nonempty_result = None
    best_nonempty_score = -1.0
    best_nonempty_total = -1
    best_nonempty_quad = None
    debug_candidates = []
    quad_eval_ranked = []
    for q in quad_candidates:
        cand = _evaluate_quad_candidate(img, q, dst_size=_board_warp_dst_size())
        score = cand["score"]
        cand_total = int(cand.get("blackCount", 0)) + int(cand.get("whiteCount", 0))
        if manual_mode:
            score *= 1.10
            if manual_reference_quad is not None:
                oq = _order_quad_points(q.astype(np.float32))
                mean_corner_delta = float(np.mean(np.abs(oq - manual_reference_quad)))
                score /= (1.0 + 0.18 * mean_corner_delta)
        # 优先级：分数优先；当分数足够接近时，用“总子更多”做 tie-break
        if best_result is None or score > best_score:
            best_score = score
            best_result = cand
            best_total = cand_total
            best_quad = _order_quad_points(q.astype(np.float32))
        elif score >= best_score * 0.97 and cand_total > best_total:
            best_score = score
            best_result = cand
            best_total = cand_total
            best_quad = _order_quad_points(q.astype(np.float32))
        debug_candidates.append(
            {
                "score": float(score),
                "blackCount": int(cand.get("blackCount", 0)),
                "whiteCount": int(cand.get("whiteCount", 0)),
                "total": int(cand_total),
            }
        )
        oq_save = _order_quad_points(q.astype(np.float32))
        quad_eval_ranked.append(
            (
                float(score),
                oq_save.copy(),
                {
                    "gridXs": [float(v) for v in (cand.get("gridXs") or [])],
                    "gridYs": [float(v) for v in (cand.get("gridYs") or [])],
                    "stones": list(cand.get("stones") or []),
                    "blackCount": int(cand.get("blackCount", 0)),
                    "whiteCount": int(cand.get("whiteCount", 0)),
                    "score": float(score),
                },
            )
        )
        if cand_total > 0:
            if best_nonempty_result is None or score > best_nonempty_score:
                best_nonempty_score = score
                best_nonempty_result = cand
                best_nonempty_total = cand_total
                best_nonempty_quad = _order_quad_points(q.astype(np.float32))
            elif score >= best_nonempty_score * 0.97 and cand_total > best_nonempty_total:
                best_nonempty_score = score
                best_nonempty_result = cand
                best_nonempty_total = cand_total
                best_nonempty_quad = _order_quad_points(q.astype(np.float32))

    # 若全局最优是空盘，但存在有子候选，优先返回有子候选，避免“未检测到棋子”。
    if best_result is not None:
        best_total = int(best_result.get("blackCount", 0)) + int(best_result.get("whiteCount", 0))
        if best_total == 0 and best_nonempty_result is not None and best_nonempty_quad is not None:
            best_result = best_nonempty_result
            best_quad = best_nonempty_quad

    if best_result is None or best_quad is None:
        fail("棋盘检测失败，请换一张更清晰或更靠近棋盘的照片")
    # 回传候选诊断，帮助定位“为何空盘/错盘胜出”
    debug_sorted = sorted(debug_candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    best_result["_debugCandidates"] = debug_sorted[: min(12, len(debug_sorted))]
    best_result["_debugSummary"] = {
        "candidateCount": int(len(debug_candidates)),
        "nonEmptyCandidateCount": int(sum(1 for x in debug_candidates if int(x.get("total", 0)) > 0)),
        "bestScore": float(best_score),
        "bestNonEmptyScore": float(best_nonempty_score if best_nonempty_score > -0.5 else -1.0),
    }
    quad_eval_ranked.sort(key=lambda t: -t[0])
    try:
        n_pack = int(os.environ.get("BOARD_ROBOFLOW_QUAD_REPICK_POOL", "14").strip() or 14)
    except Exception:
        n_pack = 14
    n_pack = max(6, min(32, n_pack))
    packs_json = []
    for sc, oq, snap in quad_eval_ranked[:n_pack]:
        gx, gy = snap.get("gridXs") or [], snap.get("gridYs") or []
        if len(gx) != 19 or len(gy) != 19:
            continue
        packs_json.append(
            {
                "score": float(sc),
                "quad": oq.astype(np.float32).tolist(),
                "gridXs": gx,
                "gridYs": gy,
                "stones": snap.get("stones") or [],
                "blackCount": int(snap.get("blackCount", 0)),
                "whiteCount": int(snap.get("whiteCount", 0)),
            }
        )
    best_result["_quadRankedPacks"] = packs_json
    return best_result, best_quad


def build_board_preview(image_bytes: bytes):
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import base64
    except Exception:
        fail("缺少依赖：请安装 python 包 opencv-python 和 numpy")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        fail("无法解析图片，请确认是有效 jpg/jpeg 文件")

    img = _downscale_bgr_for_recognition(img)

    manual_quad_raw = _read_manual_quad_from_env()
    if manual_quad_raw is not None:
        mq = np.array(manual_quad_raw, dtype=np.float32)
        if mq.shape == (4, 2):
            # 预览阶段优先使用“用户刚点击的四角”直接拉正，保证秒级反馈
            best_quad = _order_quad_points(mq)
        else:
            _, best_quad = _resolve_best_quad(img)
    else:
        _, best_quad = _resolve_best_quad(img)
    dst_size = _board_warp_dst_size()
    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(best_quad, dst)
    warped = cv2.warpPerspective(img, m, (dst_size, dst_size))

    # 在“拉正后的标准棋盘”上做一次棋子识别，用于可视化预览
    preview_eval = _evaluate_quad_candidate(img, best_quad, dst_size=dst_size)
    preview = warped.copy()
    xs = preview_eval.get("gridXs") or []
    ys = preview_eval.get("gridYs") or []
    if len(xs) != 19 or len(ys) != 19:
        margin = int(round(dst_size * 0.08))
        step_fb = (dst_size - 2 * margin) / 18.0
        xs = [int(round(margin + i * step_fb)) for i in range(19)]
        ys = [int(round(margin + i * step_fb)) for i in range(19)]

    x_left, x_right = int(xs[0]), int(xs[-1])
    y_top, y_bot = int(ys[0]), int(ys[-1])
    for yi in ys:
        yi = int(yi)
        cv2.line(preview, (x_left, yi), (x_right, yi), (80, 255, 80), 1, cv2.LINE_AA)
    for xi in xs:
        xi = int(xi)
        cv2.line(preview, (xi, y_top), (xi, y_bot), (80, 255, 80), 1, cv2.LINE_AA)

    for i in [3, 9, 15]:
        for j in [3, 9, 15]:
            cv2.circle(preview, (int(xs[i]), int(ys[j])), 4, (0, 220, 255), -1, cv2.LINE_AA)

    step_vis = float(
        max(
            8.0,
            0.5 * (float(np.median(np.diff(np.array(xs, dtype=np.float32)))) + float(np.median(np.diff(np.array(ys, dtype=np.float32))))),
        )
    )

    # 叠加识别到的黑白棋子（圆心必须与识别使用的 gridXs/gridYs 交叉点一致）
    for s in preview_eval["stones"]:
        gx = int(s["x"])
        gy = int(s["y"])
        color = s.get("color", "")
        cx = int(xs[gx])
        cy = int(ys[gy])
        r = max(8, int(round(step_vis * 0.34)))
        if color == "B":
            cv2.circle(preview, (cx, cy), r, (32, 32, 32), -1, cv2.LINE_AA)
            cv2.circle(preview, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)
        elif color == "W":
            cv2.circle(preview, (cx, cy), r, (240, 240, 240), -1, cv2.LINE_AA)
            cv2.circle(preview, (cx, cy), r, (35, 35, 35), 2, cv2.LINE_AA)

    label = f"B:{preview_eval['blackCount']}  W:{preview_eval['whiteCount']}"
    cv2.rectangle(preview, (12, 12), (200, 44), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(preview, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    ok, png = cv2.imencode(".png", preview)
    if not ok:
        fail("预览图生成失败")
    data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    return {
        "previewImageData": data_url,
        "boardSize": 19,
        "blackCount": preview_eval["blackCount"],
        "whiteCount": preview_eval["whiteCount"],
        "stones": preview_eval["stones"],
    }


def detect_board_and_stones(image_bytes: bytes):
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        fail("缺少依赖：请安装 python 包 opencv-python 和 numpy")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        fail("无法解析图片，请确认是有效 jpg/jpeg 文件")

    img = _downscale_bgr_for_recognition(img)

    # 四角评估非常耗时：整图识别只跑一次，供 Roboflow 与最终传统结果共用（避免原先 Roboflow 失败后再跑第二次）
    best_result, best_quad = _resolve_best_quad(img)
    board_pack_dl = (best_result, best_quad)

    # 深度学习优先分支：若配置了模型，优先使用 SOTA 管线；失败时自动回退传统算法。
    use_dl_first = os.environ.get("BOARD_USE_DL_FIRST", "1").strip() != "0"
    local_dl_first = os.environ.get("BOARD_LOCAL_DL_BEFORE_ROBOFLOW", "").strip() == "1"
    if use_dl_first:
        if local_dl_first:
            dl_result = _try_ultralytics_go_recognition(img, board_pack_dl)
            if dl_result is not None:
                return dl_result

        # 先尝试 Roboflow hosted API（只需要 API key，无需下载 .pt）
        use_robo = os.environ.get("BOARD_USE_ROBOFLOW_FIRST", "1").strip() != "0"
        if use_robo:
            try:
                dl_result = _try_roboflow_go_positions_recognition(
                    image_bytes, img, board_pack=board_pack_dl
                )
                if dl_result is not None:
                    return dl_result
            except Exception as e:
                _roboflow_set_last_error(f"detect_board_roboflow_exc:{e!r}")
                try:
                    import traceback

                    traceback.print_exc()
                except Exception:
                    pass

        # 再尝试本地 Ultralytics 权重（默认在 Roboflow 之后，避免重复请求云端）
        if not local_dl_first:
            dl_result = _try_ultralytics_go_recognition(img, board_pack_dl)
            if dl_result is not None:
                return dl_result

    ds_out = dict(best_result.get("_debugSummary") or {})
    ds_out["inferenceSource"] = "traditional_only"
    ds_out["roboflowKeyPresent"] = _roboflow_api_key_set()
    if _ROBOFLOW_LAST_ERROR:
        ds_out["roboflowError"] = _ROBOFLOW_LAST_ERROR

    return {
        "stones": best_result["stones"],
        "boardSize": 19,
        "blackCount": best_result["blackCount"],
        "whiteCount": best_result["whiteCount"],
        "debugCandidates": best_result.get("_debugCandidates", []),
        "debugSummary": ds_out,
    }


def main():
    image_bytes = read_input_bytes()
    preview_only = os.environ.get("BOARD_PREVIEW_ONLY", "").strip() == "1"
    result = build_board_preview(image_bytes) if preview_only else detect_board_and_stones(image_bytes)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(str(exc))
        sys.exit(1)
