const http = require('http');
const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const { Buffer } = require('buffer');

const HTTP_PORT = process.env.HTTP_PORT ? parseInt(process.env.HTTP_PORT, 10) : 5001;

/**
 * 棋盘识别：Python `board_recognizer.py`（Roboflow 云端 + 传统 CV + 可选本地 Ultralytics）。
 * - Roboflow：`ROBOFLOW_API_KEY`、`BOARD_ROBOFLOW_MODEL_ENDPOINT`（勿把密钥写入仓库）。
 * - 本地 YOLO（不依赖 Roboflow）：`BOARD_DL_MODEL` 指向 .pt/.onnx；仅黑白检测的权重配合
 *   `BOARD_DL_HAS_BOARD_CLASS=0` 与 `BOARD_DL_CLASS_MAP`；四角默认用脚本内传统搜索结果
 *  （`BOARD_DL_USE_TRADITIONAL_QUAD=1`）。若 Roboflow 常失败可先试本地：`BOARD_LOCAL_DL_BEFORE_ROBOFLOW=1`。
 * 开源参考见 `board_recognizer.py` 中 `_try_ultralytics_go_recognition` 文档字符串。
 */
const ROBOFLOW_MODEL_ENDPOINT_DEFAULT = 'synthetic-data-3ol2y/go-positions/model/6';
const RECOGNIZE_JOB_TTL_MS = 30 * 60 * 1000;
const recognizeJobs = new Map();

function stripWrappedEnvQuotes(s) {
    let t = String(s || '').trim();
    for (let i = 0; i < 5; i++) {
        if (
            t.length >= 2 &&
            ((t[0] === '"' && t[t.length - 1] === '"') ||
                (t[0] === "'" && t[t.length - 1] === "'"))
        ) {
            t = t.slice(1, -1).trim();
        } else {
            break;
        }
    }
    return t.replace(/^\/+|\/+$/g, '');
}

// KataGo 本体与资源目录（跨平台）
// Windows: C:\katago；Linux/macOS：优先 ~/KataGo（与本仓库同目录的常见布局），其次 ~/Downloads/katago、~/.katago
// 可用环境变量 KATAGO_DIR 覆盖
function defaultKatagoDir() {
    if (process.platform === 'win32') {
        return path.join('C:', 'katago');
    }
    const home = os.homedir();
    const homeKataGo = path.join(home, 'KataGo');
    const downloadsKatago = path.join(home, 'Downloads', 'katago');
    const dotKatago = path.join(home, '.katago');
    try {
        if (fs.existsSync(homeKataGo) && fs.statSync(homeKataGo).isDirectory()) {
            return homeKataGo;
        }
        if (fs.existsSync(downloadsKatago) && fs.statSync(downloadsKatago).isDirectory()) {
            return downloadsKatago;
        }
        if (fs.existsSync(dotKatago) && fs.statSync(dotKatago).isDirectory()) {
            return dotKatago;
        }
    } catch (_) {
        /* ignore */
    }
    return homeKataGo;
}
const KATAGO_DIR = process.env.KATAGO_DIR || defaultKatagoDir();

/** 在目录内选择可用的 GTP 配置（本仓库通常只有 gtp_example.cfg） */
function defaultKatagoCfg(katagoDir) {
    const def = path.join(katagoDir, 'gtp_cuda.cfg');
    const example = path.join(katagoDir, 'gtp_example.cfg');
    try {
        if (fs.existsSync(def)) return def;
        if (fs.existsSync(example)) return example;
    } catch (_) {
        /* ignore */
    }
    return def;
}

// 可通过环境变量覆盖文件名/路径，避免不同模型/配置导致频繁改代码
const KATAGO_EXE = process.env.KATAGO_EXE || (process.platform === 'win32' ? 'katago.exe' : 'katago');
const KATAGO_EXE_PATH = process.env.KATAGO_EXE_PATH || path.join(KATAGO_DIR, KATAGO_EXE);
const KATAGO_CFG = process.env.KATAGO_CFG || defaultKatagoCfg(KATAGO_DIR);
/** 未设置 KATAGO_MODEL 且请求未带 katagoModel 时使用的默认权重文件名（位于 KATAGO_DIR） */
const KATAGO_DEFAULT_MODEL_BASENAME =
    process.env.KATAGO_DEFAULT_MODEL_BASENAME || 'kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz';
const KATAGO_MODEL =
    process.env.KATAGO_MODEL || path.join(KATAGO_DIR, KATAGO_DEFAULT_MODEL_BASENAME);

/** 人机每手 kata-genmove_analyze 默认访问次数（网页 ?katagoVisits= 可覆盖；亦可通过本环境变量调整） */
const KATAGO_DEFAULT_VISITS = (() => {
    const v = parseInt(String(process.env.KATAGO_DEFAULT_VISITS || '1000').trim(), 10);
    return Number.isFinite(v) ? Math.max(15, Math.min(5000, v)) : 1000;
})();

/** 从 GTP 配置文件读取与棋力相关的项，供 /health 展示 */
function readKatagoCfgStrengthHints(cfgPath) {
    const hints = {};
    const keys = [
        'numSearchThreads',
        'maxVisits',
        'nnMaxBatchSize',
        'nnCacheSizePowerOfTwo',
        'ponderingEnabled'
    ];
    try {
        const text = fs.readFileSync(cfgPath, 'utf8');
        for (const k of keys) {
            const m = new RegExp(`^${k}\\s*=\\s*(\\S+)`, 'mi').exec(text);
            if (m) hints[k] = m[1];
        }
    } catch (_) {
        /* ignore */
    }
    return hints;
}

function katagoDirRealpath() {
    try {
        return fs.realpathSync.native ? fs.realpathSync.native(KATAGO_DIR) : fs.realpathSync(KATAGO_DIR);
    } catch (_) {
        return path.resolve(KATAGO_DIR);
    }
}

/** nginx 反代 /app2/* 时 pathname 可能带前缀，统一为 /health、/gtp、/board-recognize 等 */
function normalizeApiPathname(pathname) {
    const p = String(pathname || '/');
    const m = /^\/app\d+(?:\/(.*))?$/i.exec(p);
    if (m) {
        const rest = m[1] ? `/${m[1]}` : '/';
        return rest.replace(/\/{2,}/g, '/') || '/';
    }
    return p;
}

/** 外网/ngrok 识别：缩短 Roboflow 超时，避免中间层 ~100s 断连 */
function isFastRemoteRecognizeHost(req) {
    const host = String(req.headers.host || '');
    const xfHost = String(req.headers['x-forwarded-host'] || '');
    const combined = `${host} ${xfHost} ${req.url || ''}`;
    return /ngrok/i.test(combined) || /^\/app\d+\//i.test(String(req.url || ''));
}

/**
 * 运行 `katago version`，根据官方输出判断当前可执行文件是否为 CUDA 构建（与 GTP 实际使用的 NN 后端一致）。
 * @returns {Promise<{ ok: boolean, usesCuda: boolean, backend: string, cudaCompileVersion: string | null, raw: string }>}
 */
function probeKatagoVersionBackend() {
    return new Promise((resolve) => {
        let out = '';
        let child;
        try {
            child = spawn(KATAGO_EXE_PATH, ['version'], {
                cwd: KATAGO_DIR,
                stdio: ['ignore', 'pipe', 'pipe']
            });
        } catch (err) {
            resolve({
                ok: false,
                usesCuda: false,
                backend: 'spawn 失败',
                cudaCompileVersion: null,
                raw: String(err && err.message ? err.message : err)
            });
            return;
        }
        const append = (d) => {
            out += String(d || '');
        };
        child.stdout.on('data', append);
        child.stderr.on('data', append);
        const timer = setTimeout(() => {
            try {
                child.kill();
            } catch (_) {
                /* ignore */
            }
        }, 15000);
        const finish = () => {
            clearTimeout(timer);
            const t = out;
            let usesCuda = false;
            let backend = '未知';
            if (/using\s+cuda\s+backend/i.test(t)) {
                usesCuda = true;
                backend = 'CUDA';
            } else if (/using\s+opencl\s+backend/i.test(t)) {
                backend = 'OpenCL';
            } else if (/using\s+tensorrt\s+backend/i.test(t)) {
                backend = 'TensorRT';
            } else if (/using\s+eigen\s+backend/i.test(t)) {
                backend = 'Eigen/CPU';
            }
            const m = t.match(/compiled with cuda version\s+(\S+)/i);
            const cudaCompileVersion = m ? m[1] : null;
            resolve({ ok: true, usesCuda, backend, cudaCompileVersion, raw: t.trim() });
        };
        child.on('error', (err) => {
            clearTimeout(timer);
            resolve({
                ok: false,
                usesCuda: false,
                backend: '探测失败',
                cudaCompileVersion: null,
                raw: `${out}${String(err && err.message ? err.message : err)}`
            });
        });
        child.on('close', finish);
    });
}

/**
 * 解析客户端选择的权重路径：须在 KATAGO_DIR 内、且为 .bin.gz；空串表示使用服务器默认 KATAGO_MODEL。
 */
function resolveKatagoModelPath(clientModel) {
    const raw = String(clientModel || '').trim();
    const baseDefault = path.normalize(
        path.isAbsolute(KATAGO_MODEL) ? KATAGO_MODEL : path.resolve(KATAGO_DIR, KATAGO_MODEL)
    );
    if (!raw) {
        if (!fs.existsSync(baseDefault)) {
            throw new Error(`默认权重不存在: ${baseDefault}`);
        }
        return baseDefault;
    }
    const candidate = path.normalize(path.isAbsolute(raw) ? raw : path.join(KATAGO_DIR, raw));
    const dirReal = katagoDirRealpath();
    const rel = path.relative(dirReal, candidate);
    if (rel.startsWith('..') || path.isAbsolute(rel)) {
        throw new Error('权重路径必须在 KATAGO_DIR 目录内');
    }
    const low = candidate.toLowerCase();
    if (!low.endsWith('.bin.gz')) {
        throw new Error('权重文件须为 .bin.gz');
    }
    if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) {
        throw new Error('权重文件不存在或不是普通文件');
    }
    return candidate;
}

function listKatagoModelsInDir() {
    const out = [];
    let names = [];
    try {
        names = fs.readdirSync(KATAGO_DIR);
    } catch (_) {
        return out;
    }
    for (const name of names) {
        if (!name || typeof name !== 'string') continue;
        if (!name.toLowerCase().endsWith('.bin.gz')) continue;
        const full = path.join(KATAGO_DIR, name);
        try {
            if (fs.statSync(full).isFile()) out.push(name);
        } catch (_) {
            /* skip */
        }
    }
    out.sort((a, b) => a.localeCompare(b));
    return out;
}

/**
 * kata-genmove_analyze 的完整 GTP 响应应在最终 play 行后以空行结束（\n\n）。
 * 若仅在出现 play 行时就 resolve，可能截断后续尚未到达缓冲区的 rootInfo/winrate。
 */
function extractCompleteKataGenmoveAnalyzeBody(raw) {
    if (!raw || typeof raw !== 'string') return null;
    const s = raw.replace(/\r\n/g, '\n');
    let m = s.match(/^([\s\S]*\n(?:=\s*)?play\s+[^\n]+)\n\s*\n$/);
    if (m) return m[1].trimEnd();
    m = s.match(/^((?:=\s*)?play\s+[^\n]+)\n\s*\n$/);
    if (m) return m[1].trimEnd();
    return null;
}

function isAnalyzeCommand(command) {
    if (!command || typeof command !== 'string') return false;
    const cmd = command.trim().toLowerCase();
    return cmd.startsWith('kata-genmove_analyze') || cmd.startsWith('kata-analyze');
}

/**
 * 多盘共用一个引擎队列时，长时间命令会阻塞其它会话的 play。
 * 出队时优先处理非此类命令；排队过久的长命令仍会执行（避免永远不下完）。
 */
function isLowPriorityGtpCommand(command) {
    const low = String(command || '').trim().toLowerCase();
    if (low.startsWith('kata-genmove_analyze')) return true;
    if (low.startsWith('kata-analyze')) return true;
    if (low.startsWith('genmove')) return true;
    return false;
}

/** undo / 批量 resync 与 play 一样须优先于长时间 genmove/analyze */
function isHighPriorityGtpJob(job) {
    if (!job) return false;
    if (job.resyncBoard) return true;
    if (job.fastUndo !== undefined && job.fastUndo !== null) return true;
    return !isLowPriorityGtpCommand(job.command);
}

function countSessionPlayCommands(commands) {
    return (commands || []).filter((c) => String(c).trim().toLowerCase().startsWith('play ')).length;
}

/** 批量 clear + 重放时的单条 GTP 超时（毫秒） */
const GTP_RESYNC_CMD_TIMEOUT_MS = 60000;
const GTP_FAST_UNDO_TIMEOUT_MS = 30000;

/** analyze/genmove 超时：120 visits 在 GPU 繁忙或多盘并行时可能 >2 分钟 */
function gtpCommandTimeoutMs(command) {
    const low = String(command || '').trim().toLowerCase();
    if (!isLowPriorityGtpCommand(command)) {
        return 120000;
    }
    const capRaw = process.env.KATAGO_ANALYZE_TIMEOUT_MS;
    const cap =
        capRaw !== undefined && String(capRaw).trim() !== ''
            ? Math.max(120000, parseInt(capRaw, 10) || 600000)
            : 600000;
    let visits = 80;
    const m = /(?:kata-genmove_analyze|kata-analyze)\s+\S+\s+(\d+)/i.exec(String(command || ''));
    if (m) visits = Math.max(1, parseInt(m[1], 10) || visits);
    return Math.min(cap, Math.max(180000, visits * 4000));
}

/**
 * kata-analyze 通常返回多行 info，GTP 结束符为一个空行（\n\n）。
 * 不能按普通命令在首个 "=" 就提前结束，否则会丢失后续 info/winrate。
 * 若命令含 ownership true，须等 ownership 浮点数量足够（默认 19×19），避免流式首包就截断。
 */
function countOwnershipFloatsInAnalyzeText(text) {
    const masked = String(text || '')
        .replace(/\bmovesOwnershipStdev\b/gi, '__MOS__')
        .replace(/\bmovesOwnership\b/gi, '__MO__');
    const m = /\bownership\s+/i.exec(masked);
    if (!m) return 0;
    let rest = masked.slice(m.index + m[0].length);
    const stdevIdx = /\bownershipStdev\b/i.exec(rest);
    if (stdevIdx) rest = rest.slice(0, stdevIdx.index);
    const floats = rest.match(/-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g);
    return floats ? floats.length : 0;
}

function extractCompleteKataAnalyzeBody(raw, command) {
    if (!raw || typeof raw !== 'string') return null;
    const s = raw.replace(/\r\n/g, '\n');
    if (!/\binfo\b/i.test(s)) return null;
    if (!/\n\s*\n$/.test(s)) return null;
    const wantsOwnership = /\bownership\s+true\b/i.test(String(command || ''));
    if (wantsOwnership) {
        const need = parseInt(process.env.KATAGO_OWNERSHIP_CELLS || '361', 10) || 361;
        if (countOwnershipFloatsInAnalyzeText(s) < need) return null;
    }
    return s.trimEnd();
}

function finishAnalyzeCommand(server, currentCommand, fullResult) {
    currentCommand.completed = true;
    if (currentCommand.timeout) {
        console.log(`[DEBUG] 清除分析命令超时: ${currentCommand.command}`);
        clearTimeout(currentCommand.timeout);
        currentCommand.timeout = null;
    }
    const completedCommand = server.commandQueue.shift();
    completedCommand.resolve(fullResult);
    server.isProcessing = false;
    server.currentResponse = '';
    server.analyzeLineBuffer = [];
    server.analyzeStdoutBuffer = '';
    server.processNextCommand();
}

function readRequestBody(req, options = {}) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let totalBytes = 0;
        const maxBytes = Number.isFinite(options.maxBytes) ? options.maxBytes : 50 * 1024 * 1024;
        req.on('data', (chunk) => {
            totalBytes += chunk.length;
            if (totalBytes > maxBytes) {
                reject(new Error(`请求体过大（>${maxBytes} bytes）`));
                req.destroy();
                return;
            }
            chunks.push(chunk);
        });
        req.on('end', () => {
            resolve(Buffer.concat(chunks).toString('utf8'));
        });
        req.on('error', reject);
    });
}

function extractBase64FromDataUrl(dataUrl) {
    if (typeof dataUrl !== 'string') return null;
    // 不用整串正则匹配超长 Base64（部分手机浏览器上会失败）；支持 charset 等可选片段
    const s = dataUrl.trim();
    const marker = ';base64,';
    const i = s.indexOf(marker);
    if (i === -1) return null;
    const head = s.slice(0, i).toLowerCase();
    if (!head.startsWith('data:image/')) return null;
    return s.slice(i + marker.length).replace(/\s+/g, '');
}

function normalizeManualQuad(raw) {
    if (!Array.isArray(raw) || raw.length !== 4) return null;
    const out = [];
    for (const p of raw) {
        if (!p || typeof p !== 'object') return null;
        const x = Number(p.x);
        const y = Number(p.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        out.push({ x, y });
    }
    return out;
}

function createRecognizeJob() {
    const id = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
    recognizeJobs.set(id, {
        status: 'pending',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        result: null,
        error: null
    });
    return id;
}

function purgeRecognizeJobs() {
    const now = Date.now();
    for (const [id, job] of recognizeJobs.entries()) {
        if (!job || now - (job.updatedAt || job.createdAt || now) > RECOGNIZE_JOB_TTL_MS) {
            recognizeJobs.delete(id);
        }
    }
}

/** 仅当进程环境 BOARD_RECOGNIZE_FAST=1 时注入；不覆盖已在 env 中显式设置的变量。 */
function applyBoardRecognizeFastDefaults(env) {
    if (String(process.env.BOARD_RECOGNIZE_FAST || '').trim() !== '1') return;
    const setIfUnset = (key, value) => {
        if (stripWrappedEnvQuotes(env[key]) !== '') return;
        env[key] = value;
    };
    setIfUnset('BOARD_ROBOFLOW_INFERENCE_HOST', 'https://detect.roboflow.com');
    setIfUnset('BOARD_ROBOFLOW_MAX_INFER_SIDE', '1920');
    setIfUnset('BOARD_MAX_QUAD_EVALUATIONS', '14');
    setIfUnset('BOARD_ROBOFLOW_QUAD_REPICK_MAX', '8');
    setIfUnset('BOARD_ROBOFLOW_QUAD_REPICK_POOL', '12');
    setIfUnset('BOARD_ROBOFLOW_JPEG_QUALITY', '82');
}

function runBoardRecognizer(imageBuffer, manualQuad = null, previewOnly = false, options = {}) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'board_recognizer.py');
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        const env = { ...process.env };
        applyBoardRecognizeFastDefaults(env);
        const fastRemoteMode = !!options.fastRemoteMode && !previewOnly;
        const rfKey = stripWrappedEnvQuotes(process.env.ROBOFLOW_API_KEY);
        if (rfKey) {
            env.ROBOFLOW_API_KEY = rfKey;
        }
        const epClean = stripWrappedEnvQuotes(env.BOARD_ROBOFLOW_MODEL_ENDPOINT);
        env.BOARD_ROBOFLOW_MODEL_ENDPOINT = epClean || ROBOFLOW_MODEL_ENDPOINT_DEFAULT;
        if (fastRemoteMode) {
            // ngrok 免费隧道长请求易在约 100s 左右被中间层断开：外网自动识别启用更短的 HTTP 推理超时与单主机探测
            env.BOARD_ROBOFLOW_HTTP_TIMEOUT = String(
                Math.min(25, Math.max(12, parseInt(env.BOARD_ROBOFLOW_HTTP_TIMEOUT || '25', 10) || 25))
            );
            if (!stripWrappedEnvQuotes(env.BOARD_ROBOFLOW_INFERENCE_HOST)) {
                env.BOARD_ROBOFLOW_INFERENCE_HOST = 'https://detect.roboflow.com';
            }
        }
        if (manualQuad) {
            env.BOARD_MANUAL_QUAD = JSON.stringify(manualQuad);
        }
        if (previewOnly) {
            env.BOARD_PREVIEW_ONLY = '1';
        }
        const py = spawn(pythonCmd, [scriptPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env
        });

        let stdout = '';
        let stderr = '';
        let settled = false;
        const rawTimeout = process.env.BOARD_RECOGNIZER_TIMEOUT_MS;
        const maxMs =
            rawTimeout && String(rawTimeout).trim() !== ''
                ? Math.max(120000, parseInt(rawTimeout, 10) || 840000)
                : 840000;
        const killTimer = setTimeout(() => {
            if (settled) return;
            settled = true;
            try {
                py.kill();
            } catch (_) {}
            reject(
                new Error(
                    `识别脚本超时（>${Math.round(maxMs / 60000)} 分钟已终止）。请缩小图片或检查网络；可在环境变量 BOARD_RECOGNIZER_TIMEOUT_MS 中调整`
                )
            );
        }, maxMs);

        py.stdout.on('data', (d) => {
            stdout += d.toString();
        });
        py.stderr.on('data', (d) => {
            stderr += d.toString();
        });
        py.on('error', (err) => {
            if (settled) return;
            settled = true;
            clearTimeout(killTimer);
            reject(err);
        });
        py.on('close', (code) => {
            if (settled) return;
            settled = true;
            clearTimeout(killTimer);
            if (code !== 0) {
                reject(new Error((stderr || `识别进程退出码 ${code}`).trim()));
                return;
            }
            try {
                const parsed = JSON.parse(stdout || '{}');
                resolve(parsed);
            } catch (err) {
                reject(new Error(`识别输出解析失败: ${err.message}`));
            }
        });

        py.stdin.write(imageBuffer);
        py.stdin.end();
    });
}

class KataGoEngine {
    constructor(options = {}) {
        this.engineLabel = options.engineLabel ? String(options.engineLabel) : '';
        this.modelPath = options.modelPath
            ? path.normalize(String(options.modelPath))
            : path.normalize(KATAGO_MODEL);
        this.process = null;
        this.queue = [];
        this.busy = false;
        this.starting = null;
        this.lastReadyAt = 0;
    }

    async ensureStarted() {
        if (this.process && !this.process.killed) return;
        if (this.starting) return this.starting;

        this.starting = new Promise((resolve, reject) => {
            const tag = this.engineLabel ? ` ${this.engineLabel}` : '';
            console.log(`[KataGo] 启动引擎${tag}... model=${this.modelPath}`);
            this.process = spawn(KATAGO_EXE_PATH, ['gtp', '-config', KATAGO_CFG, '-model', this.modelPath], {
                cwd: KATAGO_DIR,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let settled = false;
            const onFail = (err) => {
                if (settled) return;
                settled = true;
                this.starting = null;
                try { if (this.process) this.process.kill(); } catch (_) {}
                this.process = null;
                reject(err);
            };

            this.process.stderr.on('data', (d) => {
                console.error(`[KataGo STDERR] ${String(d).trim()}`);
            });
            this.process.on('error', onFail);
            this.process.on('close', (code) => {
                if (!settled && code !== 0) {
                    onFail(new Error(`KataGo 进程异常退出: ${code}`));
                    return;
                }
                this.process = null;
            });

            // 通过 protocol_version 检测就绪（最多 3 分钟）
            this._sendRaw('protocol_version', 180000)
                .then(() => {
                    if (settled) return;
                    settled = true;
                    this.starting = null;
                    this.lastReadyAt = Date.now();
                    const tag = this.engineLabel ? ` ${this.engineLabel}` : '';
                    console.log(`[KataGo] 引擎${tag} 已就绪`);
                    resolve();
                })
                .catch(onFail);
        });

        return this.starting;
    }

    _sendRaw(command, timeoutMs = 30000) {
        return new Promise((resolve, reject) => {
            if (!this.process || this.process.killed || !this.process.stdin || !this.process.stdout) {
                reject(new Error('KataGo 进程不可用'));
                return;
            }
            const isAnalyze = isAnalyzeCommand(command);
            let buffer = '';
            const timer = setTimeout(() => {
                this.process.stdout.removeListener('data', onData);
                reject(new Error('命令超时'));
            }, timeoutMs);

            const done = (result) => {
                clearTimeout(timer);
                this.process.stdout.removeListener('data', onData);
                resolve(result);
            };

            const onData = (d) => {
                buffer += String(d || '');
                if (isAnalyze) {
                    const norm = buffer.replace(/\r\n/g, '\n');
                    const complete = command.toLowerCase().includes('kata-genmove_analyze')
                        ? extractCompleteKataGenmoveAnalyzeBody(norm)
                        : extractCompleteKataAnalyzeBody(norm, command);
                    if (complete != null) done(complete);
                    return;
                }
                const norm = buffer.replace(/\r\n/g, '\n');
                const lines = norm.split('\n').map(s => s.trim()).filter(Boolean);
                const resp = lines.find(l => l.startsWith('=') || l.startsWith('?'));
                if (resp) done(resp);
            };

            this.process.stdout.on('data', onData);
            this.process.stdin.write(`${command}\n`);
        });
    }

    send(command, timeoutMs = 30000) {
        return new Promise((resolve, reject) => {
            this.queue.push({ command, timeoutMs, resolve, reject });
            this._drain();
        });
    }

    async _drain() {
        if (this.busy) return;
        const item = this.queue.shift();
        if (!item) return;
        this.busy = true;
        try {
            await this.ensureStarted();
            const r = await this._sendRaw(item.command, item.timeoutMs);
            item.resolve(r);
        } catch (e) {
            item.reject(e);
        } finally {
            this.busy = false;
            this._drain();
        }
    }

    shutdown() {
        try { if (this.process) this.process.kill(); } catch (_) {}
        this.process = null;
        while (this.queue.length) {
            const q = this.queue.shift();
            try { q.reject(new Error('引擎已关闭')); } catch (_) {}
        }
    }
}

function extractMoveVertexFromResult(resultText) {
    const s = String(resultText || '');
    const m1 = /\bplay\s+(?:[BW]\s+)?([A-T][0-9]{1,2}|pass)\b/i.exec(s);
    if (m1) return m1[1].toUpperCase();
    const m2 = /^\s*=\s*([A-T][0-9]{1,2}|pass)\s*$/im.exec(s);
    if (m2) return m2[1].toUpperCase();
    return null;
}

function normalizeGtpColorToken(token) {
    const t = String(token || '').trim().toUpperCase();
    if (t === 'B' || t === 'BLACK') return 'B';
    if (t === 'W' || t === 'WHITE') return 'W';
    return null;
}

function extractColorFromCommand(commandText, re) {
    const m = re.exec(String(commandText || ''));
    return m ? normalizeGtpColorToken(m[1]) : null;
}

/** 任一行以 GTP 错误前缀 ? 开头则视为失败（play / 同步等可据此返回 success: false） */
function gtpEngineResultLooksSuccessful(resultText) {
    const lines = String(resultText || '').split(/\r?\n/);
    for (const line of lines) {
        const t = line.trim();
        if (t.startsWith('?')) return false;
    }
    return true;
}

class SessionReplayCoordinator {
    constructor(engine) {
        this.engine = engine;
        this.sessions = new Map(); // sessionId -> { commands: string[], lastAccessAt: number }
        this.activeSessionId = null;
        /** @type {Array<{ sid: string, command: string, resolve: Function, reject: Function, enqueuedAt: number }>} */
        this.gtpWaitQueue = [];
        this.gtpDrainRunning = false;
        this.starveGenmoveMs = process.env.KATAGO_GENMOVE_STARVE_MS
            ? Math.max(2000, parseInt(process.env.KATAGO_GENMOVE_STARVE_MS, 10))
            : 15000;
    }

    getSessionState(sessionId) {
        const sid = String(sessionId || 'default').slice(0, 64) || 'default';
        let state = this.sessions.get(sid);
        if (!state) {
            state = { commands: [], lastAccessAt: Date.now() };
            this.sessions.set(sid, state);
        }
        state.lastAccessAt = Date.now();
        return { sid, state };
    }

    async switchToSession(targetSessionId) {
        if (this.activeSessionId === targetSessionId) return;
        const state = this.sessions.get(targetSessionId) || { commands: [] };
        await this.engine.send('clear_board', 120000);
        for (const cmd of state.commands) {
            await this.engine.send(cmd, 120000);
        }
        this.activeSessionId = targetSessionId;
    }

    /**
     * 丢弃会话：从 replay 状态删除，并移除队列中该 sid 的待执行 GTP（否则 recordMutation 会再次 getSessionState 把会话建回来）。
     */
    forgetSessionCompletely(sid) {
        const target = String(sid || 'default').slice(0, 64) || 'default';
        const nextQueue = [];
        for (const job of this.gtpWaitQueue) {
            if (job && job.sid === target) {
                try {
                    job.reject(new Error('session_ended'));
                } catch (_) {
                    /* ignore */
                }
            } else if (job) {
                nextQueue.push(job);
            }
        }
        this.gtpWaitQueue = nextQueue;
        this.sessions.delete(target);
        if (this.activeSessionId === target) {
            this.activeSessionId = null;
        }
    }

    recordMutation(sessionId, command, result) {
        const sid = String(sessionId || 'default').slice(0, 64) || 'default';
        const state = this.sessions.get(sid);
        /** 会话已 forget 但队列里旧任务刚执行完时，勿再 getSessionState 把会话加回来 */
        if (!state) return;
        const text = String(command || '').trim();
        const low = text.toLowerCase();
        if (low === 'clear_board') {
            if (gtpEngineResultLooksSuccessful(result)) state.commands = [];
            return;
        }
        if (low.startsWith('fixed_handicap ')) {
            if (gtpEngineResultLooksSuccessful(result)) state.commands = [text];
            return;
        }
        if (low.startsWith('play ')) {
            if (gtpEngineResultLooksSuccessful(result)) state.commands.push(text);
            return;
        }
        if (low === 'undo') {
            if (gtpEngineResultLooksSuccessful(result) && state.commands.length > 0) {
                const last = String(state.commands[state.commands.length - 1] || '').trim().toLowerCase();
                if (last.startsWith('play ')) state.commands.pop();
            }
            return;
        }
        if (low.startsWith('kata-genmove_analyze ')) {
            if (!gtpEngineResultLooksSuccessful(result)) return;
            const color = extractColorFromCommand(text, /kata-genmove_analyze\s+([^\s]+)/i);
            const v = extractMoveVertexFromResult(result);
            if (color && v) state.commands.push(`play ${color} ${v}`);
            return;
        }
        if (low.startsWith('genmove ')) {
            if (!gtpEngineResultLooksSuccessful(result)) return;
            const color = extractColorFromCommand(text, /genmove\s+([^\s]+)/i);
            const v = extractMoveVertexFromResult(result);
            if (color && v) state.commands.push(`play ${color} ${v}`);
            return;
        }
        // 对于 boardsize/komi 等全局设置，按需落到会话中（避免切会话后丢规则）
        if (low.startsWith('boardsize ') || low.startsWith('komi ')) {
            if (gtpEngineResultLooksSuccessful(result)) state.commands.push(text);
        }
    }

    /**
     * 单次 HTTP 内完成 clear_board + 让子 + 全部 play，避免悔棋/还原时 N 次往返。
     * @param {string} sid
     * @param {{ handicap?: number, moves?: Array<{color:string,coord:string}> }} payload
     */
    async resyncBoardInPlace(sid, payload) {
        const { state } = this.getSessionState(sid);
        await this.switchToSession(sid);

        const newCommands = [];
        const clearR = await this.engine.send('clear_board', GTP_RESYNC_CMD_TIMEOUT_MS);
        if (!gtpEngineResultLooksSuccessful(clearR)) {
            throw new Error('clear_board 失败');
        }

        const handicap = parseInt(payload && payload.handicap, 10) || 0;
        let fixedHandicapResult = null;
        if (handicap >= 2 && handicap <= 9) {
            const fhCmd = `fixed_handicap ${handicap}`;
            const fhR = await this.engine.send(fhCmd, GTP_RESYNC_CMD_TIMEOUT_MS);
            if (!gtpEngineResultLooksSuccessful(fhR)) {
                throw new Error(`fixed_handicap ${handicap} 失败`);
            }
            newCommands.push(fhCmd);
            fixedHandicapResult = fhR;
        }

        const moves = Array.isArray(payload && payload.moves) ? payload.moves : [];
        for (const m of moves) {
            const color = normalizeGtpColorToken(m && m.color);
            if (!color) continue;
            let coord = String((m && m.coord) || '').trim();
            if (!coord) continue;
            if (coord.toUpperCase() === 'PASS') coord = 'pass';
            const cmd = `play ${color} ${coord}`;
            const pr = await this.engine.send(cmd, GTP_RESYNC_CMD_TIMEOUT_MS);
            if (!gtpEngineResultLooksSuccessful(pr)) {
                throw new Error(`同步失败: ${cmd}`);
            }
            newCommands.push(cmd);
        }

        state.commands = newCommands;
        this.activeSessionId = sid;
        return {
            playPly: countSessionPlayCommands(newCommands),
            fixedHandicapResult
        };
    }

    /**
     * 悔棋快路径：引擎与 session 手数一致时单行 undo，否则由客户端 fallback 批量 resync。
     */
    async fastUndoInPlace(sid, expectedPlayCount) {
        const expected = parseInt(expectedPlayCount, 10);
        if (!Number.isFinite(expected) || expected < 0) {
            return { success: false, mismatch: true, playPly: 0 };
        }
        const { state } = this.getSessionState(sid);
        await this.switchToSession(sid);
        const playCount = countSessionPlayCommands(state.commands);
        if (playCount !== expected + 1) {
            return { success: false, mismatch: true, playPly: playCount };
        }
        const result = await this.engine.send('undo', GTP_FAST_UNDO_TIMEOUT_MS);
        const ok = gtpEngineResultLooksSuccessful(result);
        if (ok && state.commands.length > 0) {
            const last = String(state.commands[state.commands.length - 1] || '').trim().toLowerCase();
            if (last.startsWith('play ')) state.commands.pop();
        }
        if (!ok) {
            return { success: false, mismatch: false, playPly: playCount, result };
        }
        return { success: true, playPly: expected, result };
    }

    dequeueNextGtpJob() {
        const q = this.gtpWaitQueue;
        if (q.length === 0) return null;
        const now = Date.now();
        // 必须先于「饥饿的 genmove/analyze」处理 play/clear_board/undo/resync 等短命令。
        // 否则多盘共用一个引擎时，多个排队已久的长思考会排在人类落子之前，
        // 单次 play 可能等待数分钟，触发前端 fetch 超时 →「落子失败」。
        const fastIdx = q.findIndex((it) => isHighPriorityGtpJob(it));
        if (fastIdx >= 0) return q.splice(fastIdx, 1)[0];
        for (let i = 0; i < q.length; i++) {
            const it = q[i];
            if (isLowPriorityGtpCommand(it.command) && now - it.enqueuedAt >= this.starveGenmoveMs) {
                return q.splice(i, 1)[0];
            }
        }
        return q.shift();
    }

    async runGtpDrainLoop() {
        if (this.gtpDrainRunning) return;
        this.gtpDrainRunning = true;
        try {
            while (this.gtpWaitQueue.length > 0) {
                const job = this.dequeueNextGtpJob();
                if (!job) break;
                try {
                    if (job.resyncBoard) {
                        const result = await this.resyncBoardInPlace(job.sid, job.resyncBoard);
                        job.resolve(result);
                    } else if (job.fastUndo !== undefined && job.fastUndo !== null) {
                        const result = await this.fastUndoInPlace(job.sid, job.fastUndo);
                        job.resolve(result);
                    } else {
                        await this.switchToSession(job.sid);
                        const timeoutMs = gtpCommandTimeoutMs(job.command);
                        const result = await this.engine.send(job.command, timeoutMs);
                        this.recordMutation(job.sid, job.command, result);
                        job.resolve(result);
                    }
                } catch (e) {
                    try {
                        job.reject(e);
                    } catch (_) {}
                }
            }
        } finally {
            this.gtpDrainRunning = false;
            if (this.gtpWaitQueue.length > 0) {
                void this.runGtpDrainLoop();
            }
        }
    }

    // 串行执行：切会话 + 命令；多会话时短命令优先于 genmove/analyze，避免一盘思考堵死另一盘落子
    handleGtp(sessionId, command) {
        return new Promise((resolve, reject) => {
            const { sid } = this.getSessionState(sessionId);
            this.gtpWaitQueue.push({
                sid,
                command,
                resolve,
                reject,
                enqueuedAt: Date.now()
            });
            void this.runGtpDrainLoop();
        });
    }

    handleResyncBoard(sessionId, payload) {
        return new Promise((resolve, reject) => {
            const { sid } = this.getSessionState(sessionId);
            this.gtpWaitQueue.push({
                sid,
                resyncBoard: payload,
                resolve,
                reject,
                enqueuedAt: Date.now()
            });
            void this.runGtpDrainLoop();
        });
    }

    handleFastUndo(sessionId, expectedPlayCount) {
        return new Promise((resolve, reject) => {
            const { sid } = this.getSessionState(sessionId);
            this.gtpWaitQueue.push({
                sid,
                fastUndo: expectedPlayCount,
                resolve,
                reject,
                enqueuedAt: Date.now()
            });
            void this.runGtpDrainLoop();
        });
    }
}

/**
 * 每权重并行 KataGo 进程数。多盘同时 AI 思考时，进程数 ≥ 正在思考的盘数 才能真并行。
 * 显存粗算（b40 + FP16）：单进程约 2–4GB。16GB 卡建议 3，24GB 建议 4。
 * - KATAGO_ENGINE_POOL_SIZE：显式指定（优先）
 * - KATAGO_VRAM_GB：未指定池大小时用 floor(显存GB/5)，如 16 → 3
 */
function resolveEnginePoolSize() {
    const poolRaw = process.env.KATAGO_ENGINE_POOL_SIZE;
    if (poolRaw !== undefined && String(poolRaw).trim() !== '') {
        return Math.max(1, parseInt(poolRaw, 10) || 1);
    }
    const vramGb = parseFloat(String(process.env.KATAGO_VRAM_GB || '').trim(), 10);
    if (Number.isFinite(vramGb) && vramGb > 0) {
        return Math.max(1, Math.min(6, Math.floor(vramGb / 5)));
    }
    return 2;
}

/**
 * 多盘并行：每个 KataGo 子进程内仍会串行执行 GTP，但不同进程可同时思考。
 * 新 sessionId 会绑定到当前会话数最少的引擎（负载均衡）。
 * 显存/GPU 占用 ≈ 单进程 × 池大小。见 resolveEnginePoolSize()；
 * 单进程或省显存可设 KATAGO_ENGINE_POOL_SIZE=1。
 */
class KataGoEnginePool {
    constructor(poolSize, modelAbsPath) {
        const n = Math.max(1, parseInt(String(poolSize), 10) || 1);
        const modelPath = path.normalize(String(modelAbsPath || KATAGO_MODEL));
        this.size = n;
        this.modelPath = modelPath;
        this.engines = [];
        this.coordinators = [];
        this.sessionToSlot = new Map();
        for (let i = 0; i < n; i++) {
            const label = n > 1 ? `${i + 1}/${n}` : '';
            const eng = new KataGoEngine({ engineLabel: label, modelPath });
            this.engines.push(eng);
            this.coordinators.push(new SessionReplayCoordinator(eng));
        }
    }

    coordinatorForSession(sid) {
        let slot = this.sessionToSlot.get(sid);
        if (slot === undefined) {
            let best = 0;
            let bestN = this.coordinators[0].sessions.size;
            for (let i = 1; i < this.size; i++) {
                const m = this.coordinators[i].sessions.size;
                if (m < bestN) {
                    bestN = m;
                    best = i;
                }
            }
            slot = best;
            this.sessionToSlot.set(sid, slot);
        }
        return this.coordinators[slot];
    }

    handleGtp(sid, command) {
        return this.coordinatorForSession(sid).handleGtp(sid, command);
    }

    handleResyncBoard(sid, payload) {
        return this.coordinatorForSession(sid).handleResyncBoard(sid, payload);
    }

    handleFastUndo(sid, expectedPlayCount) {
        return this.coordinatorForSession(sid).handleFastUndo(sid, expectedPlayCount);
    }

    forgetSession(sid) {
        this.sessionToSlot.delete(sid);
        for (const c of this.coordinators) {
            c.forgetSessionCompletely(sid);
        }
    }

    totalSessions() {
        return this.coordinators.reduce((acc, c) => acc + c.sessions.size, 0);
    }

    shutdown() {
        for (const e of this.engines) e.shutdown();
    }
}

/**
 * 按权重路径维护多套 KataGoEnginePool（每套内仍可有 KATAGO_ENGINE_POOL_SIZE 个进程负载均衡）。
 * 同一 sessionId 首次请求绑定权重后不可再换（避免与手顺重放错位）。
 */
class MultiModelKataGoPool {
    constructor(enginePoolSize, maxDistinctModels) {
        this.enginePoolSize = enginePoolSize;
        this.maxDistinctModels = Math.max(1, parseInt(String(maxDistinctModels), 10) || 1);
        /** @type {Map<string, KataGoEnginePool>} */
        this.poolsByModel = new Map();
        /** @type {Map<string, string>} */
        this.sessionModelLock = new Map();
    }

    evictOneUnusedPool() {
        for (const [k, pool] of this.poolsByModel.entries()) {
            if (pool.totalSessions() === 0) {
                pool.shutdown();
                this.poolsByModel.delete(k);
                return true;
            }
        }
        return false;
    }

    getOrCreatePool(modelKey) {
        const key = path.normalize(modelKey);
        const hit = this.poolsByModel.get(key);
        if (hit) return hit;
        while (this.poolsByModel.size >= this.maxDistinctModels) {
            if (!this.evictOneUnusedPool()) {
                throw new Error(
                    `同时加载的权重种类已达上限（${this.maxDistinctModels}）。请关闭其它对局页面或增大环境变量 KATAGO_MAX_DISTINCT_MODELS。`
                );
            }
        }
        const pool = new KataGoEnginePool(this.enginePoolSize, key);
        this.poolsByModel.set(key, pool);
        console.log(`[KataGo] 已创建权重池: ${path.basename(key)}（当前共 ${this.poolsByModel.size} 种权重）`);
        return pool;
    }

    handleGtp(sid, command, katagoModelOpt) {
        return this._poolForGtpSession(sid, katagoModelOpt).then((pool) => pool.handleGtp(sid, command));
    }

    _poolForGtpSession(sid, katagoModelOpt) {
        let modelAbs;
        try {
            modelAbs = resolveKatagoModelPath(katagoModelOpt);
        } catch (e) {
            return Promise.reject(e);
        }
        const key = path.normalize(modelAbs);
        const locked = this.sessionModelLock.get(sid);
        if (locked && path.normalize(locked) !== key) {
            return Promise.reject(
                new Error(
                    `本棋局已绑定权重「${path.basename(locked)}」，不可中途更换为「${path.basename(
                        key
                    )}」。请新开浏览器标签页或刷新页面后再选其它权重。`
                )
            );
        }
        let pool;
        try {
            pool = this.getOrCreatePool(key);
        } catch (e) {
            return Promise.reject(e);
        }
        if (!locked) this.sessionModelLock.set(sid, key);
        return Promise.resolve(pool);
    }

    handleResyncBoard(sid, payload, katagoModelOpt) {
        return this._poolForGtpSession(sid, katagoModelOpt).then((pool) =>
            pool.handleResyncBoard(sid, payload)
        );
    }

    handleFastUndo(sid, expectedPlayCount, katagoModelOpt) {
        return this._poolForGtpSession(sid, katagoModelOpt).then((pool) =>
            pool.handleFastUndo(sid, expectedPlayCount)
        );
    }

    coordinatorForSession(sid) {
        const key = this.sessionModelLock.get(sid);
        if (!key) return null;
        const pool = this.poolsByModel.get(path.normalize(key));
        return pool ? pool.coordinatorForSession(sid) : null;
    }

    forgetSession(sid) {
        this.sessionModelLock.delete(sid);
        for (const pool of this.poolsByModel.values()) {
            pool.forgetSession(sid);
        }
    }

    totalSessions() {
        let n = 0;
        for (const pool of this.poolsByModel.values()) n += pool.totalSessions();
        return n;
    }

    /** 所有已创建池中的子进程数上界（每池 size 之和） */
    totalEngineSlots() {
        let n = 0;
        for (const pool of this.poolsByModel.values()) n += pool.size;
        return n;
    }

    shutdown() {
        for (const pool of this.poolsByModel.values()) {
            pool.shutdown();
        }
        this.poolsByModel.clear();
        this.sessionModelLock.clear();
    }

    /** 供会话 TTL 清理遍历所有子池中的 SessionReplayCoordinator */
    allCoordinators() {
        const out = [];
        for (const pool of this.poolsByModel.values()) {
            for (const c of pool.coordinators) {
                out.push(c);
            }
        }
        return out;
    }
}

class KataGoReplayServer {
    constructor() {
        const enginePoolSize = resolveEnginePoolSize();
        const maxModelsRaw = process.env.KATAGO_MAX_DISTINCT_MODELS;
        const maxDistinctModels =
            maxModelsRaw !== undefined && String(maxModelsRaw).trim() !== ''
                ? Math.max(1, parseInt(maxModelsRaw, 10) || 1)
                : Math.max(3, enginePoolSize);
        this.enginePool = new MultiModelKataGoPool(enginePoolSize, maxDistinctModels);
        this.enginePoolSize = enginePoolSize;
        this.maxDistinctModels = maxDistinctModels;
        this.modelLocked = true;
        this.lockedModelBasename = KATAGO_DEFAULT_MODEL_BASENAME;
        this.maxSessions = process.env.KATAGO_MAX_SESSIONS
            ? Math.max(1, parseInt(process.env.KATAGO_MAX_SESSIONS, 10))
            : 32;
        this.sessionTtlMs = process.env.KATAGO_SESSION_TTL_MS
            ? Math.max(60 * 1000, parseInt(process.env.KATAGO_SESSION_TTL_MS, 10))
            : 60 * 60 * 1000;
        this.katagoBackendInfo = { ok: false, usesCuda: false, backend: '探测中…', cudaCompileVersion: null };

        this.startHttpServer();
        this.startSessionCleanup();

        process.on('SIGINT', () => {
            console.log('\n正在关闭服务器...');
            this.enginePool.shutdown();
            process.exit(0);
        });
    }

    getCurrentGlobalModelBasename() {
        return this.modelLocked && this.lockedModelBasename
            ? this.lockedModelBasename
            : KATAGO_DEFAULT_MODEL_BASENAME;
    }

    startSessionCleanup() {
        setInterval(() => {
            const now = Date.now();
            for (const coord of this.enginePool.allCoordinators()) {
                for (const [sid, st] of [...coord.sessions.entries()]) {
                    if (!st) continue;
                    if (now - (st.lastAccessAt || now) > this.sessionTtlMs) {
                        this.enginePool.forgetSession(sid);
                    }
                }
            }
            while (this.enginePool.totalSessions() > this.maxSessions) {
                let oldestSid = null;
                let oldestAt = Infinity;
                for (const coord of this.enginePool.allCoordinators()) {
                    for (const [sid, st] of coord.sessions.entries()) {
                        const t = st.lastAccessAt || 0;
                        if (t < oldestAt) {
                            oldestAt = t;
                            oldestSid = sid;
                        }
                    }
                }
                if (oldestSid == null) break;
                this.enginePool.forgetSession(oldestSid);
            }
        }, 60 * 1000);
    }

    startHttpServer() {
        const server = http.createServer(async (req, res) => {
            const reqStartAt = Date.now();
            const reqTag = `[HTTP ${new Date().toISOString()} ${Math.random().toString(36).slice(2, 8)}]`;
            console.log(`${reqTag} ${req.method} ${req.url} from ${req.socket.remoteAddress}`);
            purgeRecognizeJobs();

            req.on('aborted', () => {
                console.warn(`${reqTag} request aborted by client/proxy after ${Date.now() - reqStartAt}ms`);
            });
            res.on('close', () => {
                console.log(`${reqTag} response closed after ${Date.now() - reqStartAt}ms (status=${res.statusCode})`);
            });

            const allowedOrigins = ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8080', 'http://127.0.0.1:8080'];
            const origin = req.headers.origin;
            const isOriginAllowed = origin && allowedOrigins.includes(origin);
            res.setHeader('Access-Control-Allow-Origin', isOriginAllowed ? origin : '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
            res.setHeader('Access-Control-Allow-Credentials', isOriginAllowed ? 'true' : 'false');
            res.setHeader('Access-Control-Max-Age', '86400');

            if (req.method === 'OPTIONS') {
                res.writeHead(200);
                res.end();
                return;
            }

            const reqUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
            const apiPath = normalizeApiPathname(reqUrl.pathname);
            if (req.method === 'GET' && apiPath === '/health') {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: true,
                    service: 'ok',
                    sessions: this.enginePool.totalSessions(),
                    maxSessions: this.maxSessions,
                    enginePoolSize: this.enginePoolSize,
                    katagoDistinctModelPools: this.enginePool.poolsByModel.size,
                    katagoMaxDistinctModels: this.maxDistinctModels,
                    katagoDir: KATAGO_DIR,
                    katagoModelLocked: this.modelLocked,
                    katagoLockedModelBasename: this.lockedModelBasename || '',
                    katagoCurrentModelBasename: this.getCurrentGlobalModelBasename(),
                    katagoDefaultModel: path.normalize(
                        path.isAbsolute(KATAGO_MODEL) ? KATAGO_MODEL : path.resolve(KATAGO_DIR, KATAGO_MODEL)
                    ),
                    katagoDefaultVisits: KATAGO_DEFAULT_VISITS,
                    katagoEnginePoolSize: this.enginePoolSize,
                    katagoCfgBasename: path.basename(KATAGO_CFG),
                    katagoCfgStrength: readKatagoCfgStrengthHints(KATAGO_CFG),
                    katagoUsesCuda: !!this.katagoBackendInfo.usesCuda,
                    katagoBackend: this.katagoBackendInfo.backend || '未知',
                    katagoCudaCompileVersion: this.katagoBackendInfo.cudaCompileVersion || null,
                    boardRecognizeRoboflowKeySet: !!stripWrappedEnvQuotes(process.env.ROBOFLOW_API_KEY)
                }));
                return;
            }
            if (req.method === 'GET' && apiPath === '/katago-models') {
                try {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(
                        JSON.stringify({
                            success: true,
                            katagoDir: KATAGO_DIR,
                            defaultBasename: KATAGO_DEFAULT_MODEL_BASENAME,
                            modelLocked: true,
                            lockedBasename: this.getCurrentGlobalModelBasename(),
                            currentBasename: this.getCurrentGlobalModelBasename(),
                            models: [{ name: this.getCurrentGlobalModelBasename() }]
                        })
                    );
                } catch (err) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: false, error: err && err.message ? err.message : String(err) }));
                }
                return;
            }
            if (req.method === 'GET' && apiPath === '/board-recognize-result') {
                const jobId = String(reqUrl.searchParams.get('jobId') || '').trim();
                const job = recognizeJobs.get(jobId);
                if (!job) {
                    res.writeHead(404, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: false, error: 'job_not_found' }));
                    return;
                }
                if (job.status === 'pending') {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: true, status: 'pending' }));
                    return;
                }
                if (job.status === 'succeeded') {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: true, status: 'succeeded', result: job.result || {} }));
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true, status: 'failed', error: job.error || 'unknown_error' }));
                return;
            }

            /** 关页 sendBeacon 常用 GET（无 body）；POST JSON 见下方 */
            if (req.method === 'GET' && apiPath === '/gtp-forget-session') {
                const sid = String(reqUrl.searchParams.get('sessionId') || '').trim().slice(0, 64);
                if (sid) {
                    this.enginePool.forgetSession(sid);
                }
                res.writeHead(204);
                res.end();
                return;
            }

            if (req.method !== 'POST') {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Not Found' }));
                return;
            }

            if (apiPath === '/katago-lock-model') {
                res.writeHead(403, { 'Content-Type': 'application/json' });
                res.end(
                    JSON.stringify({
                        success: false,
                        error: 'model_change_disabled',
                        currentBasename: this.getCurrentGlobalModelBasename()
                    })
                );
                return;
            }

            if (apiPath === '/gtp-forget-session') {
                try {
                    const body = await readRequestBody(req, { maxBytes: 4096 });
                    const data = JSON.parse(body);
                    const sid = String(data.sessionId || '').trim().slice(0, 64);
                    if (sid) {
                        this.enginePool.forgetSession(sid);
                    }
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: true }));
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: false, error: 'invalid_json' }));
                }
                return;
            }

            if (apiPath === '/gtp') {
                try {
                    const body = await readRequestBody(req);
                    const data = JSON.parse(body);
                    const sessionId = data.sessionId;

                    /** 与 GTP 同路径，便于只反代了 /gtp 的 nginx；关页 beacon 走此分支 */
                    if (data.forgetGtpSession === true) {
                        const sidForget = String(sessionId || '').trim().slice(0, 64);
                        if (sidForget) {
                            this.enginePool.forgetSession(sidForget);
                        }
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: true, forgotSession: true }));
                        return;
                    }

                    const sid = String(sessionId || 'default').slice(0, 64) || 'default';
                    const modelBasename = this.getCurrentGlobalModelBasename();

                    if (data.resyncBoard && typeof data.resyncBoard === 'object') {
                        try {
                            const rb = data.resyncBoard;
                            const handicap = parseInt(rb.handicap, 10) || 0;
                            const moves = Array.isArray(rb.moves) ? rb.moves : [];
                            const out = await this.enginePool.handleResyncBoard(
                                sid,
                                { handicap, moves },
                                modelBasename
                            );
                            const sidShort = sid.length > 14 ? `${sid.slice(0, 14)}…` : sid;
                            console.log(
                                `[HTTP] 棋局[${sidShort}] 批量同步 ${moves.length} 手 play | handicap=${handicap}`
                            );
                            res.writeHead(200, { 'Content-Type': 'application/json' });
                            res.end(
                                JSON.stringify({
                                    success: true,
                                    playPly: out.playPly,
                                    fixedHandicapResult: out.fixedHandicapResult || null
                                })
                            );
                        } catch (error) {
                            res.writeHead(500, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify({ error: error && error.message ? error.message : String(error) }));
                        }
                        return;
                    }

                    if (data.fastUndo === true) {
                        const expectedPlayCount = parseInt(data.expectedPlayCount, 10);
                        if (!Number.isFinite(expectedPlayCount) || expectedPlayCount < 0) {
                            res.writeHead(400, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify({ error: 'invalid expectedPlayCount' }));
                            return;
                        }
                        try {
                            const out = await this.enginePool.handleFastUndo(
                                sid,
                                expectedPlayCount,
                                modelBasename
                            );
                            const sidShort = sid.length > 14 ? `${sid.slice(0, 14)}…` : sid;
                            console.log(
                                `[HTTP] 棋局[${sidShort}] fastUndo → playPly=${out.playPly} success=${!!out.success} mismatch=${!!out.mismatch}`
                            );
                            res.writeHead(200, { 'Content-Type': 'application/json' });
                            res.end(
                                JSON.stringify({
                                    success: out.success === true,
                                    mismatch: out.mismatch === true,
                                    playPly: out.playPly,
                                    result: out.result || null
                                })
                            );
                        } catch (error) {
                            res.writeHead(500, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify({ error: error && error.message ? error.message : String(error) }));
                        }
                        return;
                    }

                    const command = data.command;
                    if (!command || typeof command !== 'string') {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: '无效的命令' }));
                        return;
                    }

                    try {
                        const result = await this.enginePool.handleGtp(sid, command, modelBasename);
                        const coord = this.enginePool.coordinatorForSession(sid);
                        const st = coord ? coord.sessions.get(sid) : null;
                        const playPly = st
                            ? st.commands.filter((c) => String(c).trim().toLowerCase().startsWith('play ')).length
                            : 0;
                        const sidShort = sid.length > 14 ? `${sid.slice(0, 14)}…` : sid;
                        const modelTag = ` | 当前全局权重=${modelBasename}${this.modelLocked ? '（已锁定）' : ''}`;
                        console.log(`[HTTP] 棋局[${sidShort}] 本局手顺(累计 play) ${playPly} 手 | ${command}${modelTag}`);
                        const gtpOk = gtpEngineResultLooksSuccessful(result);
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: gtpOk, result }));
                    } catch (error) {
                        res.writeHead(500, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: error && error.message ? error.message : String(error) }));
                    }
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: '无效的 JSON 格式' }));
                }
                return;
            }

            if (apiPath === '/board-recognize-async') {
                try {
                    req.setTimeout(600000);
                    res.setTimeout(600000);
                    const body = await readRequestBody(req, { maxBytes: 100 * 1024 * 1024 });
                    console.log(`${reqTag} async body bytes=${Buffer.byteLength(body, 'utf8')}`);
                    const data = JSON.parse(body);
                    const base64Image = extractBase64FromDataUrl(data && data.imageData);
                    if (!base64Image) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: '请上传图片的 Base64 数据（data:image/...;base64,...）' }));
                        return;
                    }
                    const manualQuad = normalizeManualQuad(data && data.manualQuad);
                    const imageBuffer = Buffer.from(base64Image, 'base64');
                    const fastRemoteMode = isFastRemoteRecognizeHost(req);
                    const jobId = createRecognizeJob();
                    console.log(`${reqTag} async job created: ${jobId}, fastRemoteMode=${fastRemoteMode}`);
                    (async () => {
                        const job = recognizeJobs.get(jobId);
                        if (!job) return;
                        try {
                            const recognized = await runBoardRecognizer(imageBuffer, manualQuad, false, { fastRemoteMode });
                            job.status = 'succeeded';
                            job.result = recognized;
                            job.updatedAt = Date.now();
                            console.log(`${reqTag} async job succeeded: ${jobId}`);
                        } catch (err) {
                            job.status = 'failed';
                            job.error = err && err.message ? err.message : String(err);
                            job.updatedAt = Date.now();
                            console.error(`${reqTag} async job failed: ${jobId} | ${job.error}`);
                        }
                    })();
                    res.writeHead(202, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: true, jobId, status: 'pending' }));
                } catch (err) {
                    console.error(`${reqTag} async recognize setup failed: ${err && err.message ? err.message : err}`);
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `识别失败: ${err.message}` }));
                }
                return;
            }

            if (apiPath === '/board-recognize' || apiPath === '/board-preview') {
                try {
                    req.setTimeout(600000);
                    res.setTimeout(600000);
                    const body = await readRequestBody(req, { maxBytes: 100 * 1024 * 1024 });
                    console.log(`${reqTag} body bytes=${Buffer.byteLength(body, 'utf8')}`);
                    const data = JSON.parse(body);
                    const base64Image = extractBase64FromDataUrl(data && data.imageData);
                    if (!base64Image) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: '请上传图片的 Base64 数据（data:image/...;base64,...）' }));
                        return;
                    }
                    const manualQuad = normalizeManualQuad(data && data.manualQuad);
                    const previewOnly = apiPath === '/board-preview' || !!(data && data.previewOnly);
                    console.log(`${reqTag} recognize start previewOnly=${previewOnly} manualQuad=${Array.isArray(manualQuad) ? manualQuad.length : 0}`);

                    const imageBuffer = Buffer.from(base64Image, 'base64');
                    const fastRemoteMode = isFastRemoteRecognizeHost(req);
                    if (fastRemoteMode && !previewOnly) {
                        console.log(`${reqTag} fastRemoteMode=on host=${host}`);
                    }
                    const recognized = await runBoardRecognizer(imageBuffer, manualQuad, previewOnly, { fastRemoteMode });
                    console.log(`${reqTag} recognize finished in ${Date.now() - reqStartAt}ms`);
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: true, result: recognized }));
                } catch (err) {
                    console.error(`${reqTag} recognize failed after ${Date.now() - reqStartAt}ms: ${err && err.message ? err.message : err}`);
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `识别失败: ${err.message}` }));
                }
                return;
            }

            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not Found' }));
        });

        function getLocalIP() {
            const interfaces = os.networkInterfaces();
            for (const name of Object.keys(interfaces)) {
                for (const iface of interfaces[name]) {
                    if (iface.family === 'IPv4' && !iface.internal) return iface.address;
                }
            }
            return '127.0.0.1';
        }

        server.requestTimeout = 600000;
        server.headersTimeout = 660000;

        server.listen(HTTP_PORT, '0.0.0.0', () => {
            const ip = getLocalIP();
            console.log('============================================================');
            console.log('🚀 HTTP GTP 代理已启动');
            console.log('============================================================');
            console.log(`[KataGo] 可执行文件: ${KATAGO_EXE_PATH}`);
            console.log(`[KataGo] GTP 配置: ${KATAGO_CFG}`);
            probeKatagoVersionBackend()
                .then((p) => {
                    this.katagoBackendInfo = p;
                    if (p.ok && p.backend !== '未知') {
                        const ver = p.cudaCompileVersion ? `，编译 CUDA ${p.cudaCompileVersion}` : '';
                        console.log(
                            `[KataGo] 是否使用 CUDA 后端: ${p.usesCuda ? '是' : '否'}（NN 后端: ${p.backend}${ver}）`
                        );
                    } else {
                        console.warn(
                            `[KataGo] 是否使用 CUDA 后端: 无法从「katago version」判断（请确认 KATAGO_EXE_PATH 可运行）`
                        );
                        if (p.raw) {
                            console.warn(`[KataGo] version 输出: ${p.raw.slice(0, 800)}`);
                        }
                    }
                })
                .catch((err) => {
                    console.warn(`[KataGo] CUDA 后端探测异常: ${err && err.message ? err.message : err}`);
                });
            console.log(`[KataGo] 默认权重: ${KATAGO_DEFAULT_MODEL_BASENAME}`);
            console.log(`[KataGo] 人机默认 visits: ${KATAGO_DEFAULT_VISITS}（环境变量 KATAGO_DEFAULT_VISITS；网页 ?katagoVisits= 可覆盖）`);
            const poolHint =
                process.env.KATAGO_ENGINE_POOL_SIZE !== undefined &&
                String(process.env.KATAGO_ENGINE_POOL_SIZE).trim() !== ''
                    ? 'KATAGO_ENGINE_POOL_SIZE'
                    : process.env.KATAGO_VRAM_GB
                      ? `KATAGO_VRAM_GB=${process.env.KATAGO_VRAM_GB}`
                      : '默认 2（16GB 多盘建议 start_server_multigame.bat 或 KATAGO_ENGINE_POOL_SIZE=3）';
            console.log(`[KataGo] 每权重引擎池: ${this.enginePoolSize} 个进程（${poolHint}；省显存可设 1）`);
            console.log(
                `[KataGo] 最多同时保留 ${this.maxDistinctModels} 种不同权重（KATAGO_MAX_DISTINCT_MODELS）；权重列表 GET /katago-models`
            );
            if (this.enginePoolSize > 1) {
                console.log(
                    `[KataGo] 同一权重下多盘 session 将负载均衡到不同进程并行思考；显存约「单权重 × 每权重进程数 × 已加载权重种类」`
                );
            }
            console.log(`💻 本机访问:    http://localhost:${HTTP_PORT}/gtp`);
            console.log(`🌐 局域网访问:  http://${ip}:${HTTP_PORT}/gtp`);
            console.log(`[board-recognize] ROBOFLOW_API_KEY 传入 Python 子进程: ${process.env.ROBOFLOW_API_KEY ? '是（已设置）' : '否（未设置）'}`);
            console.log('============================================================');
        });

        server.on('error', (error) => {
            console.error(`HTTP 服务器错误: ${error.message}`);
        });
    }
}

// 启动服务器
new KataGoReplayServer();