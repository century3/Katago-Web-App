# Katago Web App

Browser-based Go (Weiqi) client backed by a local [KataGo](https://github.com/lightvector/KataGo) GTP engine.

## Features

- Play against KataGo in the browser (`go_board.html`)
- HTTP API wrapping KataGo GTP (`server.js`)
- Optional board photo recognition via Python (`board_recognizer.py`)
- Windows launch scripts for strong single-game or multi-game GPU use

## Requirements

- Node.js 16+
- KataGo binary + CUDA runtime (Windows) or OpenCL/CUDA build for your OS
- A KataGo neural net (`.bin.gz`), placed next to the engine
- Python 3 (only if you use board recognition)

Large binaries and network weights are **not** included in this repo. Download them separately and keep them in your KataGo directory (default on Windows: `C:\katago`).

**KataGo binary download URL:** https://github.com/lightvector/KataGo  
(Releases page for prebuilt engines: https://github.com/lightvector/KataGo/releases)

## Quick start

```bash
npm install
```

Configure paths via environment variables if needed:

| Variable | Purpose |
|---|---|
| `KATAGO_DIR` | Directory containing `katago.exe` / `katago` and models |
| `KATAGO_CFG` | GTP config (default: `gtp_cuda.cfg`) |
| `KATAGO_DEFAULT_MODEL_BASENAME` | Model filename |
| `HTTP_PORT` | API port (default `5001`) |
| `ROBOFLOW_API_KEY` | Optional, for cloud board recognition |

Windows (strong single game):

```bat
start_server_strong.bat
```

Or multi-game pool:

```bat
start_server_multigame.bat
```

Static front-end only:

```bash
node serve.js
```

Open `go_board.html` (served by `serve.js` or your API server) and point the UI at the API host/port.

## Layout

| Path | Role |
|---|---|
| `go_board.html` | Web UI |
| `server.js` | KataGo HTTP bridge + recognition |
| `serve.js` | Lightweight static file server |
| `gtp_cuda.cfg` | Example strong CUDA GTP config |
| `board_recognizer.py` | Photo → board position |
| `start_server_*.bat` | Windows launch helpers |

## License

MIT (application code). KataGo itself and neural network weights have their own licenses/terms.
